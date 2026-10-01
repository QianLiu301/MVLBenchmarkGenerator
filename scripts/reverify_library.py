"""Verify every implementation in the library again with the current rules.

    python scripts/reverify_library.py                  # dry run: verify, save, compare; writes nothing
    python scripts/reverify_library.py --apply FILE     # write a saved dry run into the database

The dry run reads all implementations in one short session and disconnects
(Neon must not be held awake), runs the full verification locally, and saves
every result to output/reverify/reverify_<time>.json together with a summary
of what would change. --apply writes such a file in one transaction; an entry
whose code no longer has the checksum the result was computed for is skipped.
Splitting the two lets the change be inspected before it reaches the site.
"""
import argparse
import collections
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv                                   # noqa: E402
load_dotenv(PROJECT_ROOT / '.env')

from sqlalchemy import select                                    # noqa: E402
from library import service                                      # noqa: E402
from library.db import session_scope                             # noqa: E402
from library.models import Benchmark, Implementation             # noqa: E402

OUT_DIR = PROJECT_ROOT / 'output' / 'reverify'


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def dry_run():
    from benchmark_validator import BenchmarkValidator
    with session_scope() as s:
        targets = [dict(id=i.id, slug=b.slug, k=b.k_value, n=b.bitwidth, language=i.language,
                        code=i.code, old_status=i.golden_status, published=i.status == 'published')
                   for i, b in s.execute(select(Implementation, Benchmark).join(Benchmark)
                                         .order_by(Implementation.id)).all()]
    print(f'{len(targets)} implementations; verifying (nothing is written)', flush=True)
    validator = quiet(BenchmarkValidator, project_root=str(PROJECT_ROOT))
    results, t0 = [], time.time()
    for n, t in enumerate(targets, 1):
        try:
            m = quiet(service.verify_code, t['code'], t['language'], t['k'], t['n'], validator)
            m['test_vectors'] = service.count_test_vectors(t['code'], t['language'])
            m['verified_at'] = m['verified_at'].isoformat()
            new = m['golden_status']
        except Exception as e:
            m, new = None, f'ERROR {type(e).__name__}: {e}'[:160]
        results.append({'id': t['id'], 'slug': t['slug'], 'language': t['language'],
                        'published': t['published'], 'old_status': t['old_status'],
                        'new_status': new, 'metrics': m})
        flag = '' if new == t['old_status'] else f'   <-- was {t["old_status"]}'
        print(f"[{n}/{len(targets)}] #{t['id']:<4d} {t['slug']:12s} {t['language']:13s} {new}{flag}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"reverify_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    path.write_text(json.dumps(results, default=str), encoding='utf-8')
    summarize(results)
    print(f'\nelapsed {(time.time() - t0) / 60:.1f} min\nsaved: {path}\n'
          f'to write it: python scripts/reverify_library.py --apply "{path}"')


def summarize(results):
    pub = [r for r in results if r['published']]
    old_pass = sum(r['old_status'] == 'PASS' for r in pub)
    new_pass = sum(r['new_status'] == 'PASS' for r in pub)
    print('\n' + '=' * 72)
    print(f'published implementations : {len(pub)}')
    print(f'verified (PASS)           : {old_pass} ({100 * old_pass / len(pub):.1f}%) -> '
          f'{new_pass} ({100 * new_pass / len(pub):.1f}%)')
    langs = sorted({r['language'] for r in pub})
    print('by language (PASS before -> after / total):')
    for lang in langs:
        rs = [r for r in pub if r['language'] == lang]
        print(f"  {lang:13s} {sum(r['old_status'] == 'PASS' for r in rs):4d} -> "
              f"{sum(r['new_status'] == 'PASS' for r in rs):4d} / {len(rs)}")
    changes = collections.Counter((r['old_status'], r['new_status']) for r in pub if r['old_status'] != r['new_status'])
    print('status changes:')
    for (a, b), c in changes.most_common():
        print(f'  {a:15s} -> {b:15s} {c:4d}')
    lost = [r for r in pub if r['old_status'] == 'PASS' and r['new_status'] != 'PASS']
    if lost:
        print('no longer verified:')
        for r in lost:
            print(f"  #{r['id']:<4d} {r['slug']:12s} {r['language']:13s} -> {r['new_status']}")
    errors = [r for r in results if r['metrics'] is None]
    if errors:
        print(f'{len(errors)} could not be verified (left unchanged by --apply)')


def apply(path: Path):
    results = json.loads(path.read_text(encoding='utf-8'))
    summarize(results)
    written = skipped = 0
    with session_scope() as s:                       # one transaction: all or nothing
        for r in results:
            m = r['metrics']
            if m is None:
                continue
            impl = s.get(Implementation, r['id'])
            if impl is None or impl.sha256 != m['sha256']:
                skipped += 1                         # deleted or changed since the dry run
                continue
            m = dict(m, verified_at=datetime.fromisoformat(m['verified_at']))
            for key, val in m.items():
                setattr(impl, key, val)
            written += 1
    print(f'\nwritten: {written}   skipped (changed or deleted since the dry run): {skipped}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--apply', metavar='FILE', help='write a saved dry run into the database')
    args = ap.parse_args()
    if args.apply:
        apply(Path(args.apply))
    else:
        dry_run()


if __name__ == '__main__':
    main()
