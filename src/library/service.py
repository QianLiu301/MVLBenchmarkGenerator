"""Library operations: slugs, ingest, verification, queries, downloads, citation."""
import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sqlalchemy import func, select

from .db import session_scope
from .models import (Benchmark, Implementation, LANGUAGES, LANGUAGE_EXT,
                     LICENSE_ID, MODULE_TYPES, SOURCES)

try:
    from galois_field import resolve_logic_type
except ImportError:  # running as package from project root
    from src.galois_field import resolve_logic_type

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CITATION = {
    'key': 'drechsler2026llmmvl',
    'authors': 'Rolf Drechsler and Qian Liu',
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


def count_test_vectors(code: str, language: str) -> int:
    inds = _TEST_INDICATORS.get(language, [])
    return sum(1 for line in code.split('\n') if any(i in line.lower() for i in inds))


def filename_for(slug: str, language: str) -> str:
    return f"{slug}{LANGUAGE_EXT.get(language, '.txt')}"


# ----------------------------------------------------------------------------
# Verification (compile → simulate → golden model), reused by seeding and submissions
# ----------------------------------------------------------------------------

def verify_code(code: str, language: str, k: int, bitwidth: int, validator=None) -> Dict:
    """Run the existing BenchmarkValidator on a code string; return metric fields."""
    if validator is None:
        try:
            from benchmark_validator import BenchmarkValidator
        except ImportError:
            from src.benchmark_validator import BenchmarkValidator
        validator = BenchmarkValidator(project_root=str(PROJECT_ROOT))

    ext = LANGUAGE_EXT.get(language, '.txt')
    # Keep verification artefacts under output/ like the rest of the pipeline
    tmp_dir = PROJECT_ROOT / 'output' / 'library_verify'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=ext, prefix=f"{make_slug('alu', k, bitwidth)}_", dir=tmp_dir)
    os.close(fd)
    try:
        Path(path).write_text(code, encoding='utf-8')
        report = validator.validate(path, k, bitwidth, language)
        summary = report.summary()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    sim_ok = summary['compile_success'] and summary['run_success']
    # Trim bulky fields; the detail page shows counts + first failures
    summary.pop('run_output', None)
    summary['failures'] = summary.get('failures', [])[:20]
    return {
        'sim_status': 'pass' if sim_ok else 'fail',
        'sim_passed': summary['passed'],
        'sim_total': summary['total_parsed'],
        'golden_status': summary['status'],
        'golden_passed': summary['passed'],
        'golden_compared': summary['total_compared'],
        'verification_report': summary,
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


def add_implementation(session, benchmark: Benchmark, language: str, code: str, *,
                       source: str = 'llm-generated', provider: str = None,
                       model_requested: str = None, model_responded: str = None,
                       metrics: Optional[Dict] = None, submitter: Optional[Dict] = None,
                       notes: str = '', status: str = 'published') -> Optional[Implementation]:
    """Insert one implementation; returns None if identical code is already stored."""
    digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
    for existing in benchmark.implementations:
        if hashlib.sha256(existing.code.encode('utf-8')).hexdigest() == digest:
            return None
    metrics = metrics or {}
    submitter = submitter or {}
    impl = Implementation(
        benchmark=benchmark,
        language=language,
        filename=filename_for(benchmark.slug, language),
        code=code,
        source=source,
        provider=provider,
        model_requested=model_requested,
        model_responded=model_responded,
        generated_at=datetime.utcnow(),
        loc=len(code.split('\n')),
        test_vectors=count_test_vectors(code, language),
        submitter_name=submitter.get('name'),
        submitter_email=submitter.get('email'),
        submitter_affiliation=submitter.get('affiliation'),
        notes=notes or '',
        status=status,
        **{k: v for k, v in metrics.items() if hasattr(Implementation, k)},
    )
    session.add(impl)
    session.flush()
    return impl


# ----------------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------------

FILTER_FIELDS = ('module_type', 'k_value', 'bitwidth', 'logic_family')


def list_benchmarks(session, filters: Dict, language: str = None, source: str = None,
                    verified_only: bool = False) -> List[Benchmark]:
    stmt = select(Benchmark).where(Benchmark.status == 'published')
    for field in FILTER_FIELDS:
        val = filters.get(field)
        if val not in (None, '', 'all'):
            col = getattr(Benchmark, field)
            stmt = stmt.where(col == (int(val) if field in ('k_value', 'bitwidth') else val))
    if language or source or verified_only:
        sub = select(Implementation.benchmark_id).where(Implementation.status == 'published')
        if language:
            sub = sub.where(Implementation.language == language)
        if source:
            sub = sub.where(Implementation.source == source)
        if verified_only:
            sub = sub.where(Implementation.golden_status == 'PASS')
        stmt = stmt.where(Benchmark.id.in_(sub))
    stmt = stmt.order_by(Benchmark.module_type, Benchmark.k_value, Benchmark.bitwidth)
    return list(session.execute(stmt).scalars().unique())


def get_benchmark(session, slug: str) -> Optional[Benchmark]:
    return session.execute(select(Benchmark).where(Benchmark.slug == slug)).scalar_one_or_none()


def facets(session) -> Dict:
    """Counts per filter value, for the sidebar and the home-page category cards."""
    out = {}
    for field in FILTER_FIELDS:
        col = getattr(Benchmark, field)
        rows = session.execute(
            select(col, func.count()).where(Benchmark.status == 'published').group_by(col).order_by(col)
        ).all()
        out[field] = [(v, n) for v, n in rows]
    lang_rows = session.execute(
        select(Implementation.language, func.count()).where(Implementation.status == 'published')
        .group_by(Implementation.language).order_by(Implementation.language)).all()
    out['language'] = [(v, n) for v, n in lang_rows]
    src_rows = session.execute(
        select(Implementation.source, func.count()).where(Implementation.status == 'published')
        .group_by(Implementation.source)).all()
    out['source'] = [(v, n) for v, n in src_rows]
    return out


def stats(session) -> Dict:
    n_bm = session.execute(select(func.count()).select_from(Benchmark)
                           .where(Benchmark.status == 'published')).scalar_one()
    n_impl = session.execute(select(func.count()).select_from(Implementation)
                             .where(Implementation.status == 'published')).scalar_one()
    n_verified = session.execute(select(func.count()).select_from(Implementation)
                                 .where(Implementation.status == 'published',
                                        Implementation.golden_status == 'PASS')).scalar_one()
    n_models = session.execute(select(func.count(func.distinct(Implementation.model_responded)))
                               .where(Implementation.status == 'published')).scalar_one()
    return {'benchmarks': n_bm, 'implementations': n_impl, 'verified': n_verified, 'models': n_models}


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
    note = f",\n  note = {{Benchmark {benchmark.slug}, {c['url']}/library/{benchmark.slug}}}" if benchmark else \
           f",\n  note = {{{c['url']}}}"
    return (f"@inproceedings{{{c['key']},\n"
            f"  author    = {{{c['authors']}}},\n"
            f"  title     = {{{c['title']}}},\n"
            f"  booktitle = {{{c['booktitle']}}},\n"
            f"  year      = {{{c['year']}}}{note}\n}}")


def spec_json(benchmark: Benchmark) -> str:
    impls = [{
        'filename': i.filename, 'language': i.language, 'source': i.source,
        'provider': i.provider, 'model': i.model_responded or i.model_requested,
        'loc': i.loc, 'test_vectors': i.test_vectors,
        'simulation': i.sim_status, 'golden_model': i.golden_status,
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
        # several implementations of the same language (different models) must not collide
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
