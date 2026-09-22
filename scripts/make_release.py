"""Build the archived data release of the library (the file that goes to Zenodo).

    python -X utf8 scripts/make_release.py            # -> output/release/mvl-benchmark-library-<version>.zip

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


def readme(release, stats, matrix, n_specs, n_impls):
    langs = ', '.join(LANGUAGES.values())
    models = '\n'.join(
        f"| {m['label']}{' (' + m['provider'] + ')' if m['provider'] else ''} | {m['verified']}/{m['total']} | {m['pct']} % |"
        for m in matrix['models'])
    return f"""# MVL Benchmark Library — release {release['version']} ({release['date']})

Reference specifications and implementations of multi-valued logic designs (arithmetic-logic
units over Z/k^nZ and GF(q)[x]/(x^n)) in {langs}, each checked against an independent
reference model and published with its verification record.

Website: https://llm-mvl.com
License: CC BY 4.0 (LICENSE.txt) — cite CITATION.bib{(' — DOI ' + release['doi']) if release['doi'] else ''}

## Contents

- {n_specs} specifications, {n_impls} implementations ({stats['verified']} verified, {stats['verified_pct']} %)
- Benchmark format v{service.FORMAT_VERSION} (https://llm-mvl.com/format)
- Reference model v{GOLDEN_MODEL_VERSION} (src/golden_model.py, src/galois_field.py of the generator repository)
- One directory per specification: `spec.json` (parameters, operations, per-file verification
  summary, SHA-256) and the implementation files. When a language has several
  implementations the file name carries the source and id, e.g. `alu_k3_8t_gemini_106.v`.
- `manifest.json`: list of specifications and files with SHA-256, format version, timestamp.

## Verification

Every file was compiled/simulated (gcc, CPython, Icarus Verilog, GHDL) and compared with the
reference model in two ways: the vectors the program prints itself (strategy A) and injected
golden vectors through a harness (strategy B, exhaustive when k^n <= 256, otherwise 50 seeded
random pairs plus edge cases). `spec.json` states the outcome and the strength label per file.
Implementations that failed are included on purpose when they come from the project's own
LLM pipeline: which model gets which specification wrong is part of the data. External
submissions are only published when they pass.

## Generation models (verified / generated)

| Model | Verified | Rate |
|---|---|---|
{models}

Full tables by language, algebraic family and failure class: https://llm-mvl.com/models
"""


def main():
    init_db()
    release = service.release_info()
    out_dir = PROJECT_ROOT / 'output' / 'release'
    out_dir.mkdir(parents=True, exist_ok=True)
    with session_scope() as s:
        benchmarks, _ = service.list_benchmarks(s, {}, sort='name', page=None)
        stats = service.stats(s)
        matrix = service.model_matrix(s)
        n_impls = sum(len(b.published_implementations) for b in benchmarks)
        data = service.build_zip(benchmarks)
        text = readme(release, stats, matrix, len(benchmarks), n_impls)

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
        'creators': [{'name': 'Drechsler, Rolf', 'affiliation': 'University of Bremen / DFKI'}],
        'description': (
            'Reference specifications and implementations of multi-valued logic designs '
            '(arithmetic-logic units over Z/k^nZ and GF(q)[x]/(x^n)) in C, Python, Verilog and VHDL, '
            f'each checked against an independent reference model (v{GOLDEN_MODEL_VERSION}) and published with '
            f'its verification record. {len(benchmarks)} specifications, {n_impls} implementations '
            f'({stats["verified"]} verified). Benchmark format v{service.FORMAT_VERSION}. '
            'Website with browsing, per-file verification logs and submission pipeline: https://llm-mvl.com'),
        'keywords': ['multi-valued logic', 'MVL', 'benchmarks', 'ALU', 'Galois field', 'hardware description',
                     'Verilog', 'VHDL', 'LLM code generation', 'verification'],
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
