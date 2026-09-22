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

CITATION_FILE = PROJECT_ROOT / 'config' / 'citation.bib'
FORMAT_VERSION = '1.0'


def citation_bibtex() -> str:
    """The BibTeX entry, byte-identical everywhere it is shown (config/citation.bib, comments stripped)."""
    try:
        text = CITATION_FILE.read_text(encoding='utf-8')
    except OSError:
        return '% config/citation.bib is missing'
    lines = [l for l in text.splitlines() if not l.lstrip().startswith('%')]
    return '\n'.join(lines).strip() + '\n'


def _citation_field(name: str) -> str:
    m = re.search(r'^\s*' + name + r'\s*=\s*\{(.*)\}\s*,?\s*$', citation_bibtex(), re.M)
    return m.group(1).strip() if m else ''


def citation_text() -> str:
    """One-line human-readable citation derived from the same file."""
    return (f"{_citation_field('author')}. \"{_citation_field('title')}\". "
            f"{_citation_field('booktitle')}, {_citation_field('year')}.")


RELEASE_FILE = PROJECT_ROOT / 'config' / 'release.json'


def release_info() -> Dict:
    """The data release (version, date, DOI) from config/release.json; DOI may be empty until minted."""
    try:
        data = json.loads(RELEASE_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    doi = (data.get('doi') or '').strip()
    return {
        'version': data.get('version') or FORMAT_VERSION,
        'date': data.get('date') or '',
        # the library's authors; the paper's author list lives in citation.bib
        'authors': data.get('authors') or [_citation_field('author')],
        'doi': doi,
        'doi_url': f"https://doi.org/{doi}" if doi else '',
        'zenodo_url': (data.get('zenodo_url') or '').strip(),
    }


def dataset_bibtex() -> str:
    """BibTeX for citing the data release itself (the Zenodo record), as opposed to the paper."""
    r = release_info()
    year = r['date'][:4] or _citation_field('year')
    lines = [
        f"@misc{{mvlbenchmarklibrary{year},",
        f"  author    = {{{' and '.join(r['authors'])}}},",
        f"  title     = {{MVL Benchmark Library, release {r['version']}}},",
        f"  year      = {{{year}}},",
        "  publisher = {Zenodo},",
    ]
    if r['doi']:
        lines.append(f"  doi       = {{{r['doi']}}},")
    lines.append("  url       = {https://llm-mvl.com}")
    return '\n'.join(lines) + '\n}\n'


LICENSE_NOTICE = (
    "MVL Benchmark Library — https://llm-mvl.com\n"
    "Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
    "https://creativecommons.org/licenses/by/4.0/\n\n"
    "You may share and adapt these files for any purpose, provided you give\n"
    "appropriate credit by citing the entry in CITATION.bib.\n"
)

# Two algebraic families (see /format §2). The reference model computes
#   modular : the ring Z/k^nZ — radix-k integer arithmetic with carry/borrow
#   field   : the ring GF(q)[x]/(x^n) — digit-wise Galois-field arithmetic, q = k a prime power
# 'prime_field' and 'integer_ring' from galois_field.py are the same arithmetic (only k differs).
_FAMILY = {'prime_field': 'modular', 'integer_ring': 'modular', 'extension_field': 'field'}
FAMILY_LABELS = {'modular': 'Z/kⁿZ — radix-k integer arithmetic (with carry)',
                 'field': 'GF(q)[x]/(xⁿ) — Galois-field polynomial arithmetic'}
_SUP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def structure_label(k: int, n: int) -> str:
    """Exact algebraic object of a spec, e.g. 'Z/3⁸Z' or 'GF(4)[x]/(x⁸)'."""
    info = resolve_logic_type(k)
    if info['category'] == 'extension_field':
        return f"GF({k})[x]/(x{str(n).translate(_SUP)})"
    return f"Z/{k}{str(n).translate(_SUP)}Z"
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
    unit = 'bit' if k == 2 else 'trit' if k == 3 else 'digit'
    title = f"{bitwidth}-{unit} {radix} {MODULE_TYPES.get(module_type, module_type)}"
    params = params or {}
    if params.get('register_count'):
        title += f" ({params['register_count']} registers)"
    if params.get('pipeline_stages'):
        title += f" ({params['pipeline_stages']}-stage)"
    ops = ', '.join(operations)
    family = _FAMILY.get(info['category'], 'modular')
    label = structure_label(k, bitwidth)
    if family == 'modular':
        semantics = (f"Words are radix-{k} integers with {bitwidth} digits (range 0 … {k ** bitwidth - 1}); "
                     f"results are reduced modulo {k ** bitwidth}, ADD/INC report a carry and SUB/DEC a borrow, "
                     f"Z = result is 0, N = result ≥ {k ** bitwidth // 2}.")
    else:
        semantics = (f"Words are polynomials of degree < {bitwidth} over GF({k}) (irreducible polynomial "
                     f"{info['tables']['irreducible_poly'] if info.get('tables') else ''}); ADD/SUB/NEG act digit-wise, "
                     f"MUL is the polynomial product truncated to {bitwidth} digits, INC/DEC add/subtract 1 in digit 0; "
                     f"carry and negative are always 0.")
    description = (f"{MODULE_TYPES.get(module_type, module_type)} over {label} with k = {k} logic values per digit "
                   f"and {bitwidth} digits per operand. Operations: {ops}. {semantics}")
    return {
        'slug': make_slug(module_type, k, bitwidth, params),
        'module_type': module_type,
        'k_value': k,
        'bitwidth': bitwidth,
        'logic_type': label,
        'logic_family': family,
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
    if b.logic_family == 'modular':
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
        # GF(q)[x]/(x^n): digit-wise in GF(q), truncated polynomial product, no carry
        # (must match GoldenModel._execute_extension and format section 2, family F)
        f = f'GF({b.k_value})'
        defs = {
            'ADD': (f'digit-wise a_i ⊕ b_i in {f}', 'carry = 0 (no carry between digits)'),
            'SUB': (f'digit-wise a_i ⊖ b_i in {f}', 'borrow = 0'),
            'MUL': (f'polynomial product a(x)·b(x) over {f}, truncated to degree < {b.bitwidth}', 'carry = 0'),
            'NEG': (f'digit-wise additive inverse in {f}', 'carry = 0'),
            'INC': ('a_0 ⊕ 1 (digit 0 only)', 'carry = 0'),
            'DEC': ('a_0 ⊖ 1 (digit 0 only)', 'borrow = 0'),
        }
    out = []
    for op in b.operations:
        formula, flag = defs.get(op, ('—', ''))
        out.append({'op': op, 'formula': formula, 'flag': flag})
    if b.logic_family == 'modular':
        zn = f'Z = 1 iff result = 0;  N = 1 iff result ≥ {half}'
    else:
        zn = 'Z = 1 iff result = 0;  N = 0 always'
    out.append({'op': 'Z / N', 'formula': zn, 'flag': ''})
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


EXHAUSTIVE_MAX_RANGE = 256   # k^n <= 256 -> all 65 536 operand pairs x 6 ops are simulated


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
    # The VHDL harness feeds vectors through textio into `integer` (32-bit); wider
    # operand ranges overflow it, so injection cannot be used there. Say so instead
    # of reporting a crash as a result.
    vhdl_range_limit = language == 'vhdl' and k ** bitwidth > 2 ** 31 - 1
    if vhdl_range_limit:
        sum_b = {'status': 'skipped', 'error': f'operand range {k}^{bitwidth} exceeds VHDL integer (2^31-1); '
                                               'injection harness not applicable',
                 'total_compared': 0, 'passed': 0, 'failed': 0}
    exhaustive = k ** bitwidth <= EXHAUSTIVE_MAX_RANGE
    if vhdl_range_limit:
        pass
    elif sum_a['compile_success'] and sum_a['run_success']:
        try:
            rep_b = validator.validate_with_injection(code, k, bitwidth, language,
                                                      random_count=random_count, seed=seed,
                                                      exhaustive=exhaustive)
            sum_b = rep_b.summary()
            if rep_b.run_output:
                log += '\n\n=== Strategy B (golden vectors via harness) ===\n' + rep_b.run_output
        except Exception as e:  # harness generation is best-effort
            sum_b = {'status': 'HARNESS_ERROR', 'error': str(e), 'total_compared': 0, 'passed': 0, 'failed': 0}

    sim_ok = sum_a['compile_success'] and sum_a['run_success']
    b_ran = bool(sum_b) and sum_b.get('total_compared', 0) > 0
    if b_ran:
        strength = (f"exhaustive(N={sum_b['total_compared']})" if exhaustive
                    else f"random(N={random_count}, seed={seed})")
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
                            'exhaustive': exhaustive,
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
IMPL_FILTERS = ('language', 'source', 'verified', 'model')


def _apply_filters(stmt, filters: Dict):
    for field in FILTER_FIELDS:
        val = filters.get(field)
        if val not in (None, '', 'all'):
            col = getattr(Benchmark, field)
            try:
                stmt = stmt.where(col == (int(val) if field in ('k_value', 'bitwidth') else val))
            except ValueError:
                pass
    language, source = filters.get('language'), filters.get('source')
    verified, model = filters.get('verified'), filters.get('model')
    if language or source or verified or model:
        sub = select(Implementation.benchmark_id).where(Implementation.status == 'published')
        if language:
            sub = sub.where(Implementation.language == language)
        if source:
            sub = sub.where(Implementation.source == source)
        if verified in ('1', 1, True, 'true'):
            sub = sub.where(Implementation.golden_status == 'PASS')
        elif verified in ('0', 0, 'false'):
            # "did not pass": the spec has an implementation that failed the check
            sub = sub.where(Implementation.golden_status != 'PASS')
        if model:
            sub = sub.where(Implementation.model_responded == model)
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


def latest_benchmarks(session, n: int = 10, module_type: str = None) -> List[Benchmark]:
    stmt = select(Benchmark).where(Benchmark.status == 'published')
    if module_type:
        stmt = stmt.where(Benchmark.module_type == module_type)
    rows = list(session.execute(stmt.order_by(Benchmark.created_at.desc(), Benchmark.id.desc()).limit(n)).scalars())
    for b in rows:
        _ = b.published_implementations
    return rows


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
    specs_u, impls_u = session.execute(
        select(func.count(func.distinct(Implementation.benchmark_id)), func.count())
        .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
        .where(Implementation.status == 'published', pub,
               Implementation.golden_status != 'PASS')).one()
    out['verified'] = [{'value': '1', 'specs': specs_v, 'impls': impls_v},
                       {'value': '0', 'specs': specs_u, 'impls': impls_u}]
    out['model'] = [{'value': v, 'specs': sp, 'impls': im} for v, sp, im in session.execute(
        select(Implementation.model_responded,
               func.count(func.distinct(Implementation.benchmark_id)), func.count())
        .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
        .where(Implementation.status == 'published', pub,
               Implementation.model_responded.isnot(None))
        .group_by(Implementation.model_responded)
        .order_by(func.count().desc())).all()]
    return out


def home_categories(session) -> Dict:
    """The two homepage panels (RevLib-style): specifications and implementations.

    Every group is a list of {label, count, url_args}; the template turns url_args
    into a link into /library, so the panels can only offer filters that exist.
    """
    f = facets(session)
    st = stats(session)
    counts = {m['value']: m['specs'] for m in module_counts(session)}
    by_value = lambda rows: {str(r['value']): r for r in rows}

    modules = []
    for value, label in MODULE_TYPES.items():
        n = counts.get(value, 0)
        modules.append({'label': label, 'count': n if n else None,
                        'note': None if n else 'planned',
                        'url_args': {'module_type': value} if n else None})

    fam = by_value(f['logic_family'])
    structure = [{'label': FAMILY_LABELS[key].split(' — ')[0],
                  'count': fam.get(key, {}).get('specs', 0),
                  'url_args': {'logic_family': key}}
                 for key in FAMILY_LABELS if key in fam]

    radix = [{'label': f"k = {r['value']}", 'count': r['specs'],
              'url_args': {'k_value': r['value']}} for r in f['k_value']]
    digits = [{'label': str(r['value']), 'count': r['specs'],
               'url_args': {'bitwidth': r['value']}} for r in f['bitwidth']]

    languages = [{'label': LANGUAGES.get(r['value'], r['value']), 'count': r['impls'],
                  'url_args': {'language': r['value']}} for r in f['language']]
    ver = by_value(f['verified'])
    verification = [{'label': 'Verified', 'count': ver.get('1', {}).get('impls', 0),
                     'url_args': {'verified': '1'}},
                    {'label': 'Did not pass', 'count': ver.get('0', {}).get('impls', 0),
                     'url_args': {'verified': '0'}}]
    sources = [{'label': SOURCES.get(r['value'], r['value']), 'count': r['impls'],
                'url_args': {'source': r['value']}} for r in f['source']]
    models = [{'label': r['value'], 'count': r['impls'], 'url_args': {'model': r['value']}}
              for r in f['model']]

    return {
        'specifications': {
            'total': st['benchmarks'],
            'groups': [{'title': 'Module', 'items': modules},
                       {'title': 'Structure', 'items': structure},
                       {'title': 'Radix k', 'items': radix, 'inline': True},
                       {'title': 'Digits', 'items': digits, 'inline': True}],
        },
        'implementations': {
            'total': st['implementations'],
            'groups': [{'title': 'Language', 'items': languages},
                       {'title': 'Verification', 'items': verification},
                       {'title': 'Source', 'items': sources},
                       {'title': 'Generation model', 'items': models}],
        },
    }


def stats(session) -> Dict:
    pub = Benchmark.status == 'published'
    n_bm = session.execute(select(func.count()).select_from(Benchmark).where(pub)).scalar_one()
    n_impl = session.execute(select(func.count()).select_from(Implementation)
                             .where(Implementation.status == 'published')).scalar_one()
    n_verified = session.execute(select(func.count()).select_from(Implementation)
                                 .where(Implementation.status == 'published',
                                        Implementation.golden_status == 'PASS')).scalar_one()
    kmin, kmax = session.execute(select(func.min(Benchmark.k_value), func.max(Benchmark.k_value)).where(pub)).one()
    dmin, dmax = session.execute(select(func.min(Benchmark.bitwidth), func.max(Benchmark.bitwidth)).where(pub)).one()
    return {
        'benchmarks': n_bm, 'implementations': n_impl, 'verified': n_verified,
        'verified_pct': round(100 * n_verified / n_impl) if n_impl else 0,
        'k_min': kmin, 'k_max': kmax, 'digits_min': dmin, 'digits_max': dmax,
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


STATUS_CLASSES = ('PASS', 'COMPILE_ERROR', 'RUNTIME_ERROR', 'LOGIC_ERROR', 'NO_OUTPUT')


def model_matrix(session) -> Dict:
    """Pass rates per generation model, for the Models page.

    One row per responding model (human-authored files form their own row); per row the
    verified/total counts by language, by algebraic family, and the golden-model status
    classes. Only published implementations of published specs are counted, including the
    failed ones the project's own pipeline keeps (they are the point of the comparison).
    """
    rows = session.execute(
        select(Implementation.source, Implementation.provider, Implementation.model_responded,
               Implementation.language, Benchmark.logic_family, Implementation.golden_status, func.count())
        .join(Benchmark, Benchmark.id == Implementation.benchmark_id)
        .where(Implementation.status == 'published', Benchmark.status == 'published')
        .group_by(Implementation.source, Implementation.provider, Implementation.model_responded,
                  Implementation.language, Benchmark.logic_family, Implementation.golden_status)).all()
    models: Dict[str, Dict] = {}
    for source, provider, model, lang, family, status, n in rows:
        if source == 'llm-generated':
            key = model or provider or 'unknown model'
            label, sub = key, provider
        else:
            key = f'source:{source}'
            label, sub = SOURCES.get(source, source), None
        m = models.setdefault(key, {
            'label': label, 'provider': sub, 'llm': source == 'llm-generated',
            'total': 0, 'verified': 0,
            'by_language': {l: [0, 0] for l in LANGUAGES},
            'by_family': {f: [0, 0] for f in FAMILY_LABELS},
            'by_status': {c: 0 for c in STATUS_CLASSES},
        })
        ok = n if status == 'PASS' else 0
        m['total'] += n; m['verified'] += ok
        if lang in m['by_language']:
            m['by_language'][lang][0] += ok; m['by_language'][lang][1] += n
        if family in m['by_family']:
            m['by_family'][family][0] += ok; m['by_family'][family][1] += n
        m['by_status'][status if status in m['by_status'] else 'NO_OUTPUT'] += n
    for m in models.values():
        m['pct'] = round(100 * m['verified'] / m['total']) if m['total'] else 0
    ordered = sorted(models.values(), key=lambda m: (not m['llm'], -m['total'], m['label']))
    return {'models': ordered, 'languages': LANGUAGES, 'families': FAMILY_LABELS,
            'statuses': STATUS_CLASSES}


def bump_downloads(session, benchmark: Benchmark = None, impl: Implementation = None):
    if benchmark is not None:
        benchmark.download_count = (benchmark.download_count or 0) + 1
    if impl is not None:
        impl.download_count = (impl.download_count or 0) + 1


# ----------------------------------------------------------------------------
# Downloads & citation
# ----------------------------------------------------------------------------

def bibtex(benchmark: Optional[Benchmark] = None) -> str:
    """config/citation.bib verbatim; for a benchmark page a `note` with its permanent URL is added.
    Archives (benchmark=None) also carry the data-release entry, with its DOI once minted."""
    text = citation_bibtex()
    if benchmark is None:
        return text + '\n' + dataset_bibtex()
    note = f"  note      = {{Benchmark {benchmark.slug}, https://llm-mvl.com/benchmark/{benchmark.slug}}}"
    body = text.rstrip().rstrip('}').rstrip()
    return body + ',\n' + note + '\n}\n'


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


API_VERSION = 1


def api_benchmark(benchmark: Benchmark, base_url: str = 'https://llm-mvl.com',
                  with_implementations: bool = True) -> Dict:
    """A benchmark as the public JSON API returns it (same fields as spec.json, plus URLs)."""
    out = {
        'slug': benchmark.slug,
        'title': benchmark.title,
        'module_type': benchmark.module_type,
        'k_value': benchmark.k_value,
        'bitwidth': benchmark.bitwidth,
        'logic_family': benchmark.logic_family,
        'structure': structure_label(benchmark.k_value, benchmark.bitwidth),
        'mod_value': benchmark.mod_value,
        'operations': benchmark.operations,
        'description': benchmark.description,
        'added': benchmark.created_at.date().isoformat() if benchmark.created_at else None,
        'url': f"{base_url}/benchmark/{benchmark.slug}",
        'spec_url': f"{base_url}/benchmark/{benchmark.slug}/spec.json",
        'download_url': f"{base_url}/benchmark/{benchmark.slug}/download",
    }
    impls = benchmark.published_implementations
    out['implementation_count'] = len(impls)
    out['verified_count'] = sum(1 for i in impls if i.golden_status == 'PASS')
    out['languages'] = sorted({i.language for i in impls})
    if with_implementations:
        out['implementations'] = [{
            'id': i.id,
            'filename': i.filename,
            'language': i.language,
            'source': i.source,
            'provider': i.provider,
            'model': i.model_responded or i.model_requested,
            'loc': i.loc,
            'sha256': i.sha256,
            'simulation': i.sim_status,
            'golden_model': i.golden_status,
            'verified': i.golden_status == 'PASS',
            'vectors_passed': i.golden_passed,
            'vectors_compared': i.golden_compared,
            'verification_strength': i.verification_strength,
            'golden_model_version': (i.verification_meta or {}).get('golden_model'),
            'license': i.license,
            'code_url': f"{base_url}/benchmark/{benchmark.slug}/{i.id}",
            'download_url': f"{base_url}/benchmark/{benchmark.slug}/{i.id}/download",
            'log_url': f"{base_url}/benchmark/{benchmark.slug}/{i.id}/log",
        } for i in impls]
    return out


def build_zip(benchmarks: Iterable[Benchmark], filters: Optional[Dict] = None,
              verified_only: bool = False, languages: Optional[set] = None) -> bytes:
    benchmarks = list(benchmarks)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('LICENSE.txt', LICENSE_NOTICE)
        zf.writestr('CITATION.bib', bibtex())
        files = []
        for bm in benchmarks:
            impls = [i for i in bm.published_implementations
                     if (not verified_only or i.golden_status == 'PASS')
                     and (not languages or i.language in languages)]
            zf.writestr(f"{bm.slug}/spec.json", spec_json(bm))
            for impl in impls:
                name = impl.filename
                same_lang = [i for i in impls if i.language == impl.language]
                if len(same_lang) > 1:
                    tag = (impl.provider or impl.source or 'x').replace('/', '-')
                    stem, ext = os.path.splitext(name)
                    name = f"{stem}_{tag}_{impl.id}{ext}"
                zf.writestr(f"{bm.slug}/{name}", impl.code)
                files.append({'path': f"{bm.slug}/{name}", 'sha256': impl.sha256, 'language': impl.language,
                              'golden_model': impl.golden_status, 'implementation_id': impl.id})
        zf.writestr('manifest.json', json.dumps({
            'format_version': FORMAT_VERSION,
            'generated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'filters': filters or {},
            'verified_only': verified_only,
            'spec_ids': [bm.slug for bm in benchmarks],
            'files': files,
            'license': LICENSE_ID,
        }, indent=2))
    return buf.getvalue()


# --- Download by selection ----------------------------------------------------
MAX_SELECTION_SPECS = 500
SELECTION_FIELDS = ('module_type', 'k_value', 'bitwidth', 'logic_family', 'language')
_ZIP_CACHE_DIR = PROJECT_ROOT / 'output' / 'zip_cache'


def normalize_selection(args) -> Dict:
    """Multi-value filters from a query string -> sorted, de-duplicated dict (cache key material)."""
    sel = {}
    for f in SELECTION_FIELDS:
        vals = sorted({v for v in args.getlist(f) if v})
        if vals:
            sel[f] = vals
    sel['verified_only'] = args.get('verified_only', '1') not in ('0', 'false', '')
    return sel


def _selection_stmt(sel: Dict):
    stmt = select(Benchmark).where(Benchmark.status == 'published')
    for f in ('module_type', 'logic_family'):
        if sel.get(f):
            stmt = stmt.where(getattr(Benchmark, f).in_(sel[f]))
    for f in ('k_value', 'bitwidth'):
        if sel.get(f):
            stmt = stmt.where(getattr(Benchmark, f).in_([int(v) for v in sel[f]]))
    sub = select(Implementation.benchmark_id).where(Implementation.status == 'published')
    if sel.get('language'):
        sub = sub.where(Implementation.language.in_(sel['language']))
    if sel.get('verified_only'):
        sub = sub.where(Implementation.golden_status == 'PASS')
    return stmt.where(Benchmark.id.in_(sub)).order_by(Benchmark.k_value, Benchmark.bitwidth)


def selection_summary(session, sel: Dict) -> Dict:
    rows = list(session.execute(_selection_stmt(sel)).scalars())
    n_files = 0
    n_bytes = 0
    for b in rows:
        for i in b.published_implementations:
            if sel.get('language') and i.language not in sel['language']:
                continue
            if sel.get('verified_only') and i.golden_status != 'PASS':
                continue
            n_files += 1
            n_bytes += len(i.code.encode('utf-8'))
    return {'specs': len(rows), 'files': n_files, 'bytes': n_bytes, 'limit': MAX_SELECTION_SPECS}


def selection_zip(session, sel: Dict) -> Tuple[Optional[bytes], Optional[str]]:
    """(zip bytes, error). Cached on disk keyed by the normalized filter set + library size."""
    rows = list(session.execute(_selection_stmt(sel)).scalars())
    if len(rows) > MAX_SELECTION_SPECS:
        return None, f"Selection has {len(rows)} specs; the limit is {MAX_SELECTION_SPECS}. Narrow the filters."
    if not rows:
        return None, "No specs match this selection."
    # cache key: filters + newest change among the selected specs (so a republish invalidates).
    # One aggregate query — iterating b.implementations would lazy-load every spec over the network.
    newest = session.execute(
        select(func.max(Implementation.created_at))
        .where(Implementation.benchmark_id.in_([b.id for b in rows]))).scalar() or datetime.min
    key = hashlib.sha256(json.dumps({'sel': sel, 'n': len(rows), 'newest': newest.isoformat()},
                                    sort_keys=True).encode()).hexdigest()[:24]
    _ZIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ZIP_CACHE_DIR / f"{key}.zip"
    if path.exists():
        return path.read_bytes(), None
    filters = {k: v for k, v in sel.items() if k != 'verified_only'}
    data = build_zip(rows, filters=filters, verified_only=sel.get('verified_only', True),
                     languages=set(sel['language']) if sel.get('language') else None)
    path.write_bytes(data)
    return data, None
