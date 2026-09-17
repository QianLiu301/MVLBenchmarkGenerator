"""Populate the library: generate with an LLM, verify against the golden model, store.

    python -X utf8 scripts/seed_library.py --provider deepseek --k 3 --bits 8 --langs c,verilog
    python -X utf8 scripts/seed_library.py --provider deepseek --k 2,3,4,5,7 --bits 8,10,12,14 --langs c,python,verilog,vhdl
    python -X utf8 scripts/seed_library.py --import-file path/to/file.v --k 3 --bits 8 --source human-authored

Everything that compiles and runs is stored, including golden-model FAILs: a
benchmark library that only shows successes hides exactly the information
(which model gets which spec wrong) that makes LLM comparison interesting. Use
--verified-only to keep PASS implementations only.
"""
import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

from benchmark_validator import BenchmarkValidator
from library import service
from library.db import init_db, session_scope
from library.models import LANGUAGE_EXT
from mvl_generator import MVLGenerator

DEFAULT_OPS = ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC']


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw), buf.getvalue()


def generate_one(provider, model, k, bits, lang, operations):
    """Return (code, call_state) or (None, error)."""
    gen = MVLGenerator(llm_provider=provider, model=model,
                       output_dir=str(PROJECT_ROOT / 'output' / 'mvl_code'))
    result = None
    for ev, data in gen.generate_stream(k_value=k, bitwidth=bits, language=lang, operations=operations):
        if ev == 'done':
            result = data
        elif ev == 'error':
            return None, data
    if not result or not result.get('success'):
        return None, (result or {}).get('error', 'no result')
    return result['code'], result


def store(session, spec, lang, code, *, source, provider, model_req, model_resp, metrics, notes=''):
    bm = service.get_or_create_benchmark(session, spec)
    impl = service.add_implementation(
        session, bm, lang, code, source=source, provider=provider,
        model_requested=model_req, model_responded=model_resp, metrics=metrics, notes=notes)
    return bm, impl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', default='deepseek')
    ap.add_argument('--model', default=None)
    ap.add_argument('--module', default='alu', choices=['alu'])
    ap.add_argument('--k', default='3', help='comma list, e.g. 2,3,4,5,7')
    ap.add_argument('--bits', default='8', help='comma list, e.g. 8,10,12,14')
    ap.add_argument('--langs', default='c,python,verilog,vhdl')
    ap.add_argument('--ops', default=','.join(DEFAULT_OPS))
    ap.add_argument('--verified-only', action='store_true', help='store only golden-model PASS')
    ap.add_argument('--import-file', help='store an existing file instead of generating')
    ap.add_argument('--source', default='llm-generated', choices=['llm-generated', 'human-authored', 'reference'])
    ap.add_argument('--notes', default='')
    ap.add_argument('--reset', action='store_true', help='delete ALL library rows first (local dev)')
    ap.add_argument('--reverify', action='store_true', help='re-run verification on every stored implementation')
    args = ap.parse_args()

    init_db()
    if args.reset:
        from library.models import Benchmark, Implementation
        with session_scope() as s:
            s.query(Implementation).delete(); s.query(Benchmark).delete()
        print('library cleared')
    validator, _ = quiet(BenchmarkValidator, project_root=str(PROJECT_ROOT))  # its tool probe is very chatty
    operations = [o.strip().upper() for o in args.ops.split(',') if o.strip()]
    ks = [int(x) for x in args.k.split(',')]
    bitss = [int(x) for x in args.bits.split(',')]
    langs = [x.strip() for x in args.langs.split(',')]

    # ---- re-verify mode ----------------------------------------------------
    if args.reverify:
        from library.models import Implementation
        with session_scope() as s:
            impls = s.query(Implementation).all()
            for i in impls:
                bm = i.benchmark
                metrics, _ = quiet(service.verify_code, i.code, i.language, bm.k_value, bm.bitwidth, validator)
                before = i.golden_status
                for key, val in metrics.items():
                    setattr(i, key, val)
                i.test_vectors = service.count_test_vectors(i.code, i.language)
                print(f"{bm.slug} {i.language:8s} #{i.id}: {before} -> {i.golden_status} "
                      f"{i.golden_passed}/{i.golden_compared}")
        return

    # ---- import mode -------------------------------------------------------
    if args.import_file:
        path = Path(args.import_file)
        lang = {v: k for k, v in LANGUAGE_EXT.items()}.get(path.suffix.lower())
        if lang is None:
            sys.exit(f"cannot infer language from {path.suffix}")
        code = path.read_text(encoding='utf-8')
        k, bits = ks[0], bitss[0]
        spec = service.describe_spec(args.module, k, bits, operations)
        metrics, log = quiet(service.verify_code, code, lang, k, bits, validator)
        print(f"{path.name}: sim={metrics['sim_status']} golden={metrics['golden_status']} "
              f"{metrics['golden_passed']}/{metrics['golden_compared']}")
        if args.verified_only and metrics['golden_status'] != 'PASS':
            sys.exit('not verified; not stored (--verified-only)')
        with session_scope() as s:
            bm, impl = store(s, spec, lang, code, source=args.source, provider=None,
                             model_req=None, model_resp=None, metrics=metrics, notes=args.notes)
            print('stored' if impl else 'identical code already stored', '->', bm.slug)
        return

    # ---- generate mode -----------------------------------------------------
    total = len(ks) * len(bitss) * len(langs)
    n = 0; stored = 0; skipped = 0; failed = 0
    t_all = time.time()
    for k in ks:
        for bits in bitss:
            spec = service.describe_spec(args.module, k, bits, operations)
            for lang in langs:
                n += 1
                t0 = time.time()
                tag = f"[{n}/{total}] {spec['slug']} {lang:8s} {args.provider}/{args.model or 'default'}"
                (code, info), log = quiet(generate_one, args.provider, args.model, k, bits, lang, operations)
                if code is None:
                    failed += 1
                    print(f"{tag}  ✗ generation failed: {info}")
                    continue
                metrics, vlog = quiet(service.verify_code, code, lang, k, bits, validator)
                verdict = f"sim={metrics['sim_status']} golden={metrics['golden_status']} " \
                          f"{metrics['golden_passed']}/{metrics['golden_compared']}"
                if args.verified_only and metrics['golden_status'] != 'PASS':
                    skipped += 1
                    print(f"{tag}  – {verdict}  (not stored)  {time.time()-t0:.0f}s")
                    continue
                with session_scope() as s:
                    bm, impl = store(s, spec, lang, code, source='llm-generated', provider=args.provider,
                                     model_req=info.get('requested_model'), model_resp=info.get('response_model'),
                                     metrics=metrics)
                if impl is None:
                    skipped += 1
                    print(f"{tag}  = identical code already stored  {time.time()-t0:.0f}s")
                else:
                    stored += 1
                    mark = '✓' if metrics['golden_status'] == 'PASS' else '!'
                    print(f"{tag}  {mark} {verdict}  model={info.get('response_model')}  {time.time()-t0:.0f}s")
    print(f"\nstored {stored}, skipped {skipped}, failed {failed} — {time.time()-t_all:.0f}s total")


if __name__ == '__main__':
    main()
