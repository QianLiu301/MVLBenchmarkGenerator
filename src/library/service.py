"""Library operations: specs, verification, ingest, queries, downloads, citation."""
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func, or_, select

from .db import session_scope
from .models import (Benchmark, Implementation, LANGUAGES, LANGUAGE_EXT,
                     LICENSE_ID, MODULE_TYPES, ReviewEvent, SOURCES,
                     GOLDEN_MODEL_VERSION)

try:
    from galois_field import resolve_logic_type
except ImportError:  # running as package from project root
    from src.galois_field import resolve_logic_type

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAGE_SIZE = 25
LOG_LIMIT = 64 * 1024

CITATION = {
    'key': 'drechsler2026llmmvl',
    'authors': 'Rolf Drechsler',
    'title': 'LLM-based Generation of High-Level Benchmarks for MVL Designs',
    'booktitle': 'IEEE International Symposium on Multiple-Valued Logic (ISMVL)',
    'year': '2026',
    'url': 'https://llm-mvl.com',
}

LICENSE_NOTICE = (
    "MVL Benchmark Library — https://llm-mvl.com\n"
    "Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
    "https://creativecommons.org/licenses/by/4.0/\n\n"
    "You may share and adapt these files for any purpose, provided you give\n"
    "appropriate credit by citing:\n"
    f"  {CITATION['authors']}. \"{CITATION['title']}\". {CITATION['booktitle']}, {CITATION['year']}.\n"
)

_FAMILY = {'prime_field': 'gf-prime', 'extension_field': 'gf-ext', 'integer_ring': 'ring'}
FAMILY_LABELS = {'gf-prime': 'GF(p) prime field', 'gf-ext': 'GF(pⁿ) extension field', 'ring': 'Z/kZ integer ring'}
RADIX_NAMES = {2: 'binary', 3: 'ternary', 4: 'quaternary', 5: 'quinary', 6: 'senary', 7: 'septenary'}

_TEST_INDICATORS = {
    'c': ['printf', 'test', 'assert'],
    'python': ['print', 'test', 'assert'],
    'verilog': ['$display', 'initial begin', '#'],
    'vhdl': ['assert', 'report', 'wait for'],
}

SORT_OPTIONS = {
    'name': (Benchmark.slug.asc(),),
    'k': (Benchmark.k_value.asc(), Benchmark.bitwidth.asc()),
    'digits': (Benchmark.bitwidth.asc(), Benchmark.k_value.asc()),
    'date': (Benchmark.created_at.desc(),),
}


# ----------------------------------------------------------------------------
# Spec helpers
# ----------------------------------------------------------------------------

def make_slug(module_type: str, k: int, bitwidth: int, params: Optional[Dict] = None) -> str:
    """Stable identifier, RevLib-style: alu_k3_8t, regfile_k3_8t_r16, cpu_k3_8t_p5."""
    base = {'alu': 'alu', 'register': 'regfile', 'cpu-risc-v': 'cpu'}.get(module_type, module_type)
    slug = f"{base}_k{k}_{bitwidth}t"
    params = params or {}
    if params.get('register_count'):
        slug += f"_r{params['register_count']}"
    if params.get('pipeline_stages'):
        slug += f"_p{params['pipeline_stages']}"
    return slug


def describe_spec(module_type: str, k: int, bitwidth: int, operations: List[str],
                  params: Optional[Dict] = None) -> Dict:
    info = resolve_logic_type(k)
    radix = RADIX_NAMES.get(k, f'radix-{k}')
    title = f"{bitwidth}-trit {radix} {MODULE_TYPES.get(module_type, module_type)}"
    params = params or {}
    if params.get('register_count'):
        title += f" ({params['register_count']} registers)"
    if params.get('pipeline_stages'):
        title += f" ({params['pipeline_stages']}-stage)"
    ops = ', '.join(operations)
    description = (f"{MODULE_TYPES.get(module_type, module_type)} over {info['display']} with k = {k} logic values "
                   f"per digit and {bitwidth} digits per operand (operand range 0 … {k ** bitwidth - 1}). "
                   f"Operations: {ops}. Results are reduced modulo {k ** bitwidth}; zero, negative and "
                   f"carry flags are reported for every operation.")
    return {
        'slug': make_slug(module_type, k, bitwidth, params),
        'module_type': module_type,
        'k_value': k,
        'bitwidth': bitwidth,
        'logic_type': info['display'],
        'logic_family': _FAMILY.get(info['category'], 'ring'),
        'mod_value': k ** bitwidth,
        'operations': list(operations),
        'params': {kk: v for kk, v in params.items() if v},
        'title': title,
        'description': description,
    }


def op_definitions(b: Benchmark) -> List[Dict]:
    """Mathematical definition of every operation, for the Spec block."""
    M = b.mod_value
    half = M // 2
    if b.logic_family == 'ring' or b.logic_family == 'gf-prime':
        # integer arithmetic modulo k^n (GF(p) with carry propagation is the same as Z/p^nZ here)
        defs = {
            'ADD': (f'(a + b) mod {M}', f'carry = 1 iff a + b ≥ {M}'),
            'SUB': (f'(a − b + {M}) mod {M}', 'borrow = 1 iff a < b'),
            'MUL': (f'(a · b) mod {M}', 'carry = 0'),
            'NEG': (f'({M} − a) mod {M}', 'carry = 0'),
            'INC': (f'(a + 1) mod {M}', f'carry = 1 iff a = {M - 1}'),
            'DEC': (f'(a − 1 + {M}) mod {M}', 'borrow = 1 iff a = 0'),
        }
    else:
        info = resolve_logic_type(b.k_value)
        p, n = info.get('p'), info.get('n')
        f = f'GF({p}^{n})'
        defs = {
            'ADD': (f'digit-wise a_i ⊕ b_i in {f}', 'carry = 0 (no carry in a field)'),
            'SUB': (f'digit-wise a_i ⊖ b_i in {f}', 'borrow = 0'),
            'MUL': (f'digit-wise a_i ⊗ b_i in {f}', 'carry = 0'),
            'NEG': (f'digit-wise additive inverse in {f}', 'carry = 0'),
            'INC': (f'digit-wise a_i ⊕ 1', f'carry = 1 iff a = {M - 1}'),
            'DEC': (f'digit-wise a_i ⊖ 1', 'borrow = 1 iff a = 0'),
        }
    out = []
    for op in b.operations:
        formula, flag = defs.get(op, ('—', ''))
        out.append({'op': op, 'formula': formula, 'flag': flag})
    out.append({'op': 'Z / N', 'formula': f'Z = 1 iff result = 0;  N = 1 iff result ≥ {half}', 'flag': ''})
    return out


def count_test_vectors(code: str, language: str) -> int:
    inds = _TEST_INDICATORS.get(language, [])
    return sum(1 for line in code.split('\n') if any(i in line.lower() for i in inds))


def filename_for(slug: str, language: str) -> str:
    return f"{slug}{LANGUAGE_EXT.get(language, '.txt')}"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ----------------------------------------------------------------------------
# Verification (compile → simulate → golden model), reused by seeding and submissions
# ----------------------------------------------------------------------------

_TOOL_VERSIONS: Optional[Dict[str, str]] = None


def tool_versions() -> Dict[str, str]:
    """Versions of the simulators actually used, recorded on every verification."""
    global _TOOL_VERSIONS
    if _TOOL_VERSIONS is not None:
        return _TOOL_VERSIONS
    probes = {
        'gcc': (['gcc', '--version'], r'(\d+\.\d+\.\d+)'),
        'python': (['python', '--version'], r'(\d+\.\d+\.\d+)'),
        'iverilog': (['iverilog', '-V'], r'Icarus Verilog version\s+([\d.]+)'),
        'ghdl': (['ghdl', '--version'], r'GHDL\s+([\d.]+)'),
    }
    out = {}
    for name, (cmd, pat) in probes.items():
        try:
            if shutil.which(cmd[0]) is None:
                continue
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, errors='replace')
            m = re.search(pat, (r.stdout or '') + (r.stderr or ''))
            out[name] = m.group(1) if m else 'present'
        except Exception:
            continue
    _TOOL_VERSIONS = out
    return out


def _trim(report: Dict) -> Dict:
    report = dict(report)
    report.pop('run_output', None)
    report['failures'] = report.get('failures', [])[:20]
    return report


def verify_code(code: str, language: str, k: int, bitwidth: int, validator=None,
                random_count: int = 50, seed: int = 42) -> Dict:
    """Run the BenchmarkValidator on a code string and return the Implementation fields.

    Strategy A: compile + run the file as-is and compare every test vector it
    prints against the golden model ("self-reported" vectors).
    Strategy B: replace the test section by a harness fed with N golden random
    vectors (deterministic seed) — this is the verification *strength* shown on
    the detail page. B is skipped when the harness cannot be built.
    """
    if validator is None:
        try:
            from benchmark_validator import BenchmarkValidator
        except ImportError:
            from src.benchmark_validator import BenchmarkValidator
        validator = BenchmarkValidator(project_root=str(PROJECT_ROOT))

    ext = LANGUAGE_EXT.get(language, '.txt')
    tmp_dir = PROJECT_ROOT / 'output' / 'library_verify'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=ext, prefix=f"{make_slug('alu', k, bitwidth)}_", dir=tmp_dir)
    os.close(fd)
    try:
        Path(path).write_text(code, encoding='utf-8')
        rep_a = validator.validate(path, k, bitwidth, language)
        log = rep_a.run_output or ''
        sum_a = rep_a.summary()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    sum_b = None
    if sum_a['compile_success'] and sum_a['run_success']:
        try:
            rep_b = validator.validate_with_injection(code, k, bitwidth, language,
                                                      random_count=random_count, seed=seed)
            sum_b = rep_b.summary()
            if rep_b.run_output:
                log += '\n\n=== Strategy B (golden vectors via harness) ===\n' + rep_b.run_output
        except Exception as e:  # harness generation is best-effort
            sum_b = {'status': 'HARNESS_ERROR', 'error': str(e), 'total_compared': 0, 'passed': 0, 'failed': 0}

    sim_ok = sum_a['compile_success'] and sum_a['run_success']
    b_ran = bool(sum_b) and sum_b.get('total_compared', 0) > 0
    if b_ran:
        strength = f"random(N={random_count}, seed={seed})"
        golden_status = 'PASS' if (sum_a['status'] == 'PASS' and sum_b['status'] == 'PASS') else \
            (sum_b['status'] if sum_b['status'] != 'PASS' else sum_a['status'])
    else:
        strength = f"self-reported(N={sum_a['total_compared']})"
        golden_status = sum_a['status']

    passed = sum_a['passed'] + (sum_b['passed'] if b_ran else 0)
    compared = sum_a['total_compared'] + (sum_b['total_compared'] if b_ran else 0)
    return {
        'sha256': sha256(code),
        'sim_status': 'pass' if sim_ok else 'fail',
        'sim_passed': sum_a['passed'],
        'sim_total': sum_a['total_parsed'],
        'golden_status': golden_status,
        'golden_passed': passed,
        'golden_compared': compared,
        'verification_strength': strength,
        'verification_meta': {
            'tools': tool_versions(),
            'golden_model': GOLDEN_MODEL_VERSION,
            'strategy_a': {'status': sum_a['status'], 'compared': sum_a['total_compared'], 'passed': sum_a['passed']},
            'strategy_b': ({'status': sum_b['status'], 'compared': sum_b.get('total_compared', 0),
                            'passed': sum_b.get('passed', 0), 'N': random_count, 'seed': seed,
                            'error': sum_b.get('error')} if sum_b else {'status': 'skipped'}),
        },
        'verification_report': {'strategy_a': _trim(sum_a), 'strategy_b': _trim(sum_b) if sum_b else None},
        'verification_log': log[-LOG_LIMIT:],
        'verified_at': datetime.utcnow(),
    }


# ----------------------------------------------------------------------------
# Ingest
# ----------------------------------------------------------------------------

def get_or_create_benchmark(session, spec: Dict) -> Benchmark:
    bm = session.execute(select(Benchmark).where(Benchmark.slug == spec['slug'])).scalar_one_or_none()
    if bm is None:
        bm = Benchmark(**spec)
        session.add(bm)
        session.flush()
    return bm


def find_duplicate(session, code: str) -> Optional[Implementation]:
    return session.execute(select(Implementation).where(Implementation.sha256 == sha256(code))).scalars().first()


def add_implementation(session, benchmark: Benchmark, language: str, code: str, *,
                       source: str = 'llm-generated', provider: str = None,
                       model_requested: str = None, model_responded: str = None,
                       prompt_hash: str = None, metrics: Optional[Dict] = None,
                       submitter: Optional[Dict] = None, notes: str = '',
                       status: str = 'published') -> Optional[Implementation]:
    """Insert one implementation; returns None if identical code is already stored."""
    digest = sha256(code)
    for existing in benchmark.implementations:
        if existing.sha256 == digest:
            return None
    metrics = metrics or {}
    submitter = submitter or {}
    impl = Implementation(
        benchmark=benchmark,
        language=language,
        filename=filename_for(benchmark.slug, language),
        code=code,
        sha256=digest,
        source=source,
        provider=provider,
        model_requested=model_requested,
        model_responded=model_responded,
        prompt_hash=prompt_hash,
        generated_at=datetime.utcnow(),
        loc=len(code.split('\n')),
        test_vectors=count_test_vectors(code, language),
        submitter_name=submitter.get('name'),
        submitter_email=submitter.get('email'),
        submitter_affiliation=submitter.get('affiliation'),
        notes=notes or '',
        status=status,
        **{k: v for k, v in metrics.items() if hasattr(Implementation, k) and k != 'sha256'},
    )
    session.add(impl)
    session.flush()
    return impl


def add_review_event(session, benchmark: Benchmark, action: str, detail: str = '', actor: str = 'system',
                     submission_id: int = None, implementation_id: int = None):
    ev = ReviewEvent(benchmark=benchmark, action=action, detail=detail, actor=actor,
                     submission_id=submission_id, implementation_id=implementation_id)
    session.add(ev)
    return ev


# ----------------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------------

FILTER_FIELDS = ('module_type', 'k_value', 'bitwidth', 'logic_family')
IMPL_FILTERS = ('language', 'source', 'verified')


def _apply_filters(stmt, filters: Dict):
    for field in FILTER_FIELDS:
        val = filters.get(field)
        if val not in (None, '', 'all'):
            col = getattr(Benchmark, field)
            try:
                stmt = stmt.where(col == (int(val) if field in ('k_value', 'bitwidth') else val))
            except ValueError:
                pass
    language, source, verified = filters.get('language'), filters.get('source'), filters.get('verified')
    if language or source or verified:
        sub = select(Implementation.benchmark_id).where(Implementation.status == 'published')
        if language:
            sub = sub.where(Implementation.language == language)
        if source:
            sub = sub.where(Implementation.source == source)
        if verified:
            sub = sub.where(Implementation.golden_status == 'PASS')
        stmt = stmt.where(Benchmark.id.in_(sub))
    q = (filters.get('q') or '').strip()
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Benchmark.slug.ilike(like), Benchmark.title.ilike(like),
                              Benchmark.logic_type.ilike(like), Benchmark.description.ilike(like)))
    return stmt


def list_benchmarks(session, filters: Dict, sort: str = 'name', page: int = None) -> Tuple[List[Benchmark], int]:
    """Published benchmarks matching filters. Returns (rows, total). page=None → all rows."""
    base = _apply_filters(select(Benchmark).where(Benchmark.status == 'published'), filters)
    total = session.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    stmt = base.order_by(*SORT_OPTIONS.get(sort, SORT_OPTIONS['name']))
    if page is not None:
        stmt = stmt.offset(max(page - 1, 0) * PAGE_SIZE).limit(PAGE_SIZE)
    rows = list(session.execute(stmt).scalars().unique())
    for b in rows:
        _ = b.published_implementations
    return rows, total


def get_benchmark(session, slug: str) -> Optional[Benchmark]:
    return session.execute(select(Benchmark).where(Benchmark.slug == slug)).scalar_one_or_none()


def recent_benchmarks(session, n: int = 5) -> List[Benchmark]:
    rows = list(session.execute(select(Benchmark).where(Benchmark.status == 'published')
                                .order_by(Benchmark.created_at.desc()).limit(n)).scalars())
    for b in rows:
        _ = b.published_implementations
    return rows


def facets(session) -> Dict:
    """Spec counts per filter value; language/source/verified also carry implementation counts."""
    out = {}
    pub = Benchmark.status == 'published'
    for field in FILTER_FIELDS:
        col = getattr(Benchmark, field)
        rows = session.execute(select(col, func.count()).where(pub).group_by(col).order_by(col)).all()
        out[field] = [{'value': v, 'specs': n} for v, n in rows]

    def dual(col, where=None):
        stmt = (select(col, func.count(func.distinct(Implementation.benchmark_id)), func.count())
                .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
                .where(Implementation.status == 'published', pub))
        if where is not None:
            stmt = stmt.where(where)
        return [{'value': v, 'specs': s, 'impls': i} for v, s, i in
                session.execute(stmt.group_by(col).order_by(col)).all()]

    out['language'] = dual(Implementation.language)
    out['source'] = dual(Implementation.source)
    specs_v, impls_v = session.execute(
        select(func.count(func.distinct(Implementation.benchmark_id)), func.count())
        .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
        .where(Implementation.status == 'published', pub, Implementation.golden_status == 'PASS')).one()
    out['verified'] = [{'value': '1', 'specs': specs_v, 'impls': impls_v}]
    return out


def stats(session) -> Dict:
    pub = Benchmark.status == 'published'
    n_bm = session.execute(select(func.count()).select_from(Benchmark).where(pub)).scalar_one()
    n_impl = session.execute(select(func.count()).select_from(Implementation)
                             .where(Implementation.status == 'published')).scalar_one()
    n_verified = session.execute(select(func.count()).select_from(Implementation)
                                 .where(Implementation.status == 'published',
                                        Implementation.golden_status == 'PASS')).scalar_one()
    kmin, kmax = session.execute(select(func.min(Benchmark.k_value), func.max(Benchmark.k_value)).where(pub)).one()
    return {
        'benchmarks': n_bm, 'implementations': n_impl, 'verified': n_verified,
        'verified_pct': round(100 * n_verified / n_impl) if n_impl else 0,
        'k_min': kmin, 'k_max': kmax,
    }


def module_counts(session) -> List[Dict]:
    rows = session.execute(select(Benchmark.module_type, func.count()).where(Benchmark.status == 'published')
                           .group_by(Benchmark.module_type)).all()
    counts = {v: n for v, n in rows}
    return [{'value': m, 'label': label, 'specs': counts.get(m, 0)} for m, label in MODULE_TYPES.items()]


def contributors(session) -> Dict:
    """People and models behind the published implementations, for the Acknowledgements page."""
    rows = session.execute(
        select(Implementation.submitter_name, Implementation.submitter_affiliation, func.count())
        .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
        .where(Implementation.status == 'published', Benchmark.status == 'published',
               Implementation.submitter_name.isnot(None))
        .group_by(Implementation.submitter_name, Implementation.submitter_affiliation)
        .order_by(Implementation.submitter_name)).all()
    people = [{'name': n, 'affiliation': a, 'count': c} for n, a, c in rows]
    models = session.execute(
        select(Implementation.provider, Implementation.model_responded, func.count())
        .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
        .where(Implementation.status == 'published', Benchmark.status == 'published',
               Implementation.source == 'llm-generated')
        .group_by(Implementation.provider, Implementation.model_responded)
        .order_by(Implementation.provider)).all()
    return {'people': people,
            'models': [{'provider': p, 'model': m, 'count': c} for p, m, c in models]}


def bump_downloads(session, benchmark: Benchmark = None, impl: Implementation = None):
    if benchmark is not None:
        benchmark.download_count = (benchmark.download_count or 0) + 1
    if impl is not None:
        impl.download_count = (impl.download_count or 0) + 1


# ----------------------------------------------------------------------------
# Downloads & citation
# ----------------------------------------------------------------------------

def bibtex(benchmark: Optional[Benchmark] = None) -> str:
    c = CITATION
    note = f",\n  note      = {{Benchmark {benchmark.slug}, {c['url']}/benchmark/{benchmark.slug}}}" if benchmark else \
           f",\n  note      = {{{c['url']}}}"
    return (f"@inproceedings{{{c['key']},\n"
            f"  author    = {{{c['authors']}}},\n"
            f"  title     = {{{c['title']}}},\n"
            f"  booktitle = {{{c['booktitle']}}},\n"
            f"  year      = {{{c['year']}}}{note}\n}}")


def spec_json(benchmark: Benchmark) -> str:
    impls = [{
        'filename': i.filename, 'language': i.language, 'source': i.source,
        'provider': i.provider, 'model': i.model_responded or i.model_requested,
        'sha256': i.sha256, 'loc': i.loc, 'test_vectors': i.test_vectors,
        'simulation': i.sim_status, 'golden_model': i.golden_status,
        'verification_strength': i.verification_strength,
        'version': i.version, 'license': i.license,
    } for i in benchmark.published_implementations]
    return json.dumps({
        'slug': benchmark.slug, 'title': benchmark.title, 'module_type': benchmark.module_type,
        'k_value': benchmark.k_value, 'bitwidth': benchmark.bitwidth,
        'logic_type': benchmark.logic_type, 'mod_value': benchmark.mod_value,
        'operations': benchmark.operations, 'params': benchmark.params,
        'description': benchmark.description, 'implementations': impls,
        'license': LICENSE_ID, 'citation': bibtex(benchmark),
    }, indent=2)


def _add_benchmark_to_zip(zf: zipfile.ZipFile, benchmark: Benchmark, prefix: str = ''):
    zf.writestr(f"{prefix}{benchmark.slug}/spec.json", spec_json(benchmark))
    for impl in benchmark.published_implementations:
        name = impl.filename
        same_lang = [i for i in benchmark.published_implementations if i.language == impl.language]
        if len(same_lang) > 1:
            tag = (impl.provider or impl.source or 'x').replace('/', '-')
            stem, ext = os.path.splitext(name)
            name = f"{stem}_{tag}_{impl.id}{ext}"
        zf.writestr(f"{prefix}{benchmark.slug}/{name}", impl.code)


def build_zip(benchmarks: Iterable[Benchmark]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('LICENSE.txt', LICENSE_NOTICE)
        zf.writestr('CITATION.bib', bibtex())
        for bm in benchmarks:
            _add_benchmark_to_zip(zf, bm)
    return buf.getvalue()
