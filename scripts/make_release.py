"""Build the archived data release of the library (the file that goes to Zenodo).

    python -X utf8 scripts/make_release.py            # -> output/release/mvl-benchmark-library-<version>.zip
    python -X utf8 scripts/make_release.py --version 1.1 --date 2026-10-02

--version/--date override config/release.json for the archive, so a release can be
built before the site is switched to its DOI. In double-blind review mode
(library/review_mode.py) the README, CITATION.bib and zenodo.json carry no author
names and nothing about the generation models.

The archive is the library-wide download (every published implementation, spec.json,
LICENSE.txt, CITATION.bib, manifest.json) plus a README.md that states the release,
the format and reference-model versions and the content counts. Next to the zip the
script writes zenodo.json with the metadata to enter on Zenodo, and prints the SHA-256
of the archive so the record can be cross-checked after upload.
"""
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / '.env')

from library import service  # noqa: E402
from library.db import init_db, session_scope  # noqa: E402
from library.models import GOLDEN_MODEL_VERSION, LANGUAGES  # noqa: E402
from library.review_mode import ANONYMOUS_REVIEW  # noqa: E402


def readme(release, stats, matrix, n_specs, n_impls, present):
    langs = ', '.join(LANGUAGES[l] for l in LANGUAGES if l in present)
    if ANONYMOUS_REVIEW:
        example = 'carries its id, e.g. `alu_k3_8t_106.v`'
        failed = ('Implementations that did not pass are included with their verification records:\n'
                  'what fails, and how, is part of the data. External submissions are only published\n'
                  'when they pass.')
        models_section = ''
    else:
        example = 'carries the source and id, e.g. `alu_k3_8t_gemini_106.v`'
        failed = ('Implementations that failed are included on purpose when they come from the project\'s own\n'
                  'LLM pipeline: which model gets which specification wrong is part of the data. External\n'
                  'submissions are only published when they pass.')
        models = '\n'.join(
            f"| {m['label']}{' (' + m['provider'] + ')' if m['provider'] else ''} | {m['verified']}/{m['total']} | {m['pct']} % |"
            for m in matrix['models'])
        models_section = f"""
## Generation models (verified / generated)

| Model | Verified | Rate |
|---|---|---|
{models}

Full tables by language, algebraic family and failure class: https://llm-mvl.com/models
"""
    return f"""# MVL Benchmark Library — release {release['version']} ({release['date']})

Reference specifications and implementations of multi-valued logic designs (arithmetic-logic
units over Z/k^nZ and GF(q)[x]/(x^n)) in {langs}, each checked against an independent
reference model and published with its verification record.

Website: https://llm-mvl.com
License: CC BY 4.0 (LICENSE.txt) — cite CITATION.bib{(' — DOI ' + release['doi']) if release['doi'] else ''}

## Contents

- {n_specs} specifications, {n_impls} implementations ({stats['verified']} verified, {stats['verified_pct']} %)
- Benchmark format v{service.FORMAT_VERSION} (https://llm-mvl.com/format)
- Reference model v{GOLDEN_MODEL_VERSION} (https://llm-mvl.com/format#reference-model)
- One directory per specification: `spec.json` (parameters, operations, per-file verification
  summary, SHA-256) and the implementation files. When a language has several
  implementations the file name {example}.
- `manifest.json`: list of specifications and files with SHA-256, format version, timestamp.

## Verification

Every file was compiled/simulated (gcc, CPython, Icarus Verilog, GHDL) and compared with the
reference model in two ways: the vectors the program prints itself (strategy A) and injected
golden vectors through a harness (strategy B, exhaustive when k^n <= 256, otherwise 50 seeded
random pairs plus edge cases). A test passes only if the result and every status flag agree.
`spec.json` states the outcome and the strength label per file.
{failed}
{models_section}"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', help='release version (default: config/release.json)')
    ap.add_argument('--date', help='release date YYYY-MM-DD (default: config/release.json)')
    args = ap.parse_args()
    init_db()
    release = service.release_info()
    if args.version:
        release['version'] = args.version
    if args.date:
        release['date'] = args.date
    if args.version or args.date:
        release['doi'] = ''          # a new release gets its own DOI when it is published
        release['doi_url'] = ''
        # the archive's CITATION.bib reads service.release_info(); make it describe this
        # release, not the one in config/release.json (whose DOI is an older record)
        service.release_info = lambda: dict(release)
    out_dir = PROJECT_ROOT / 'output' / 'release'
    out_dir.mkdir(parents=True, exist_ok=True)
    with session_scope() as s:
        benchmarks, _ = service.list_benchmarks(s, {}, sort='name', page=None)
        stats = service.stats(s)
        matrix = service.model_matrix(s)
        n_impls = sum(len(b.published_implementations) for b in benchmarks)
        present = {i.language for b in benchmarks for i in b.published_implementations}
        data = service.build_zip(benchmarks)
        text = readme(release, stats, matrix, len(benchmarks), n_impls, present)

    # add README.md at the top of the library archive
    src = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('README.md', text)
        for item in src.infolist():
            zf.writestr(item, src.read(item.filename))
    blob = buf.getvalue()

    zip_path = out_dir / f"mvl-benchmark-library-{release['version']}.zip"
    zip_path.write_bytes(blob)
    (out_dir / 'README.md').write_text(text, encoding='utf-8')
    sha = hashlib.sha256(blob).hexdigest()

    meta = {
        'upload_type': 'dataset',
        'title': f"MVL Benchmark Library — release {release['version']}",
        'creators': ([{'name': 'Anonymous'}] if ANONYMOUS_REVIEW else
                     [{'name': 'Drechsler, Rolf', 'affiliation': 'University of Bremen / DFKI'},
                      {'name': 'Liu, Qian', 'affiliation': 'University of Bremen'}]),
        'description': (
            'Reference specifications and implementations of multi-valued logic designs '
            '(arithmetic-logic units over Z/k^nZ and GF(q)[x]/(x^n)) in C, Python, Verilog and VHDL, '
            f'each checked against an independent reference model (v{GOLDEN_MODEL_VERSION}) and published with '
            f'its verification record. {len(benchmarks)} specifications, {n_impls} implementations '
            f'({stats["verified"]} verified). Benchmark format v{service.FORMAT_VERSION}. '
            'Website with browsing, per-file verification logs and submission pipeline: https://llm-mvl.com'),
        'keywords': ['multi-valued logic', 'MVL', 'benchmarks', 'ALU', 'Galois field', 'hardware description',
                     'Verilog', 'VHDL', 'verification'] + ([] if ANONYMOUS_REVIEW else ['LLM code generation']),
        'license': 'cc-by-4.0',
        'access_right': 'open',
        'version': release['version'],
        'publication_date': release['date'],
        'related_identifiers': [{'identifier': 'https://llm-mvl.com', 'relation': 'isSupplementTo', 'scheme': 'url'}],
        'language': 'eng',
        '_archive_sha256': sha,
        '_archive_file': zip_path.name,
    }
    (out_dir / 'zenodo.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"release {release['version']} ({release['date']})")
    print(f"  {zip_path}  {len(blob) / 1024:.0f} KB")
    print(f"  sha256 {sha}")
    print(f"  {len(benchmarks)} specs, {n_impls} implementations, {stats['verified']} verified")
    print(f"  metadata: {out_dir / 'zenodo.json'}")


if __name__ == '__main__':
    main()
