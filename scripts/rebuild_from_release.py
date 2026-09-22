"""Rebuild the library database from a published release archive.

    python -X utf8 scripts/rebuild_from_release.py output/release/mvl-benchmark-library-1.0.zip

The archive is the library's own published form — one directory per specification with
spec.json and the implementation files — so it is a complete description of what the
library contains. This restores a database from it when the live database is gone or is
being moved to another host. Verification logs are not part of an archive; run
`seed_library.py --reverify` afterwards to produce them again from the files themselves.

The target is whatever DATABASE_URL points at (the bundled SQLite file when it is unset).
Existing rows for a slug are left alone unless --reset is given.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / '.env')

from library import service  # noqa: E402
from library.db import init_db, session_scope  # noqa: E402
from library.models import Benchmark, Implementation  # noqa: E402

LANG_OF_EXT = {'.c': 'c', '.py': 'python', '.v': 'verilog', '.vhd': 'vhdl'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive')
    ap.add_argument('--reset', action='store_true', help='drop every row first')
    args = ap.parse_args()

    init_db()
    if args.reset:
        with session_scope() as s:
            s.query(Implementation).delete()
            s.query(Benchmark).delete()
        print('cleared')

    zf = zipfile.ZipFile(args.archive)
    specs = sorted(n for n in zf.namelist() if n.endswith('/spec.json'))
    n_specs = n_impls = 0
    for name in specs:
        spec = json.loads(zf.read(name))
        folder = name.rsplit('/', 1)[0]
        payload = {
            'slug': spec['slug'], 'module_type': spec['module_type'],
            'k_value': spec['k_value'], 'bitwidth': spec['bitwidth'],
            'logic_type': spec['logic_type'],
            'logic_family': service._FAMILY.get(
                service.resolve_logic_type(spec['k_value'])['category'], 'modular'),
            'mod_value': spec['mod_value'], 'operations': spec['operations'],
            'params': spec.get('params') or {}, 'title': spec['title'],
            'description': spec['description'],
        }
        # the archive renames files when a language has several implementations, and
        # spec.json records the plain name for all of them, so match on the checksum
        by_sha = {}
        for entry in zf.namelist():
            if entry.startswith(folder + '/') and Path(entry).suffix in LANG_OF_EXT:
                text = zf.read(entry).decode('utf-8')
                by_sha[service.sha256(text)] = text

        with session_scope() as s:
            bm = service.get_or_create_benchmark(s, payload)
            n_specs += 1
            for impl in spec['implementations']:
                code = by_sha.get(impl['sha256'])
                if code is None:
                    print(f"  ! {spec['slug']}: no file with checksum {impl['sha256'][:12]}… "
                          f"({impl['filename']})")
                    continue
                metrics = {
                    'sim_status': impl['simulation'],
                    'golden_status': impl['golden_model'],
                    'verification_strength': impl['verification_strength'],
                }
                added = service.add_implementation(
                    s, bm, impl['language'], code, source=impl['source'],
                    provider=impl['provider'], model_responded=impl['model'],
                    metrics=metrics)
                if added is not None:
                    n_impls += 1
        print(f"{spec['slug']}: {len(spec['implementations'])} implementations")
    print(f"\nrestored {n_specs} specifications, {n_impls} implementations")
    print('next: seed_library.py --reverify   (recreates the verification records)')


if __name__ == '__main__':
    main()
