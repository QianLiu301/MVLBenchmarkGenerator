"""The example on the submit page must stay a submission that passes.

Users copy it, so it is checked the way a real submission is: the manifest
against the schema, the file against the lint rules, and the code by the full
verification (the file's own tests and the injected golden vectors). No
database is touched.
"""
import io
import json
import sys
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

EXAMPLE = ROOT / 'web' / 'static' / 'examples' / 'alu_k3_8t'


def _example():
    manifest = json.loads((EXAMPLE / 'manifest.json').read_text(encoding='utf-8'))
    files = {f['filename']: (EXAMPLE / f['filename']).read_text(encoding='utf-8')
             for f in manifest['files']}
    return manifest, files


def test_example_manifest_and_lint():
    from library import submissions
    manifest, files = _example()
    assert submissions.validate_manifest(manifest, files) == []
    for f in manifest['files']:
        assert submissions.lint_file(f['filename'], files[f['filename']], f['language']) == []


def test_example_verifies():
    from library import service
    manifest, files = _example()
    for f in manifest['files']:
        with contextlib.redirect_stdout(io.StringIO()):
            m = service.verify_code(files[f['filename']], f['language'],
                                    manifest['k_value'], manifest['bitwidth'])
        assert m['golden_status'] == 'PASS', m['verification_meta']
        assert m['verification_meta']['strategy_b']['compared'] > 0


def test_wrong_flag_fails():
    """Flags are outputs: right results with a wrong borrow flag must not verify."""
    from library import service
    manifest, files = _example()
    f = manifest['files'][0]
    code = files[f['filename']]
    broken = code.replace('c = (a < b);', 'c = 1\'b0;')   # SUB never reports a borrow
    assert broken != code
    with contextlib.redirect_stdout(io.StringIO()):
        m = service.verify_code(broken, f['language'], manifest['k_value'], manifest['bitwidth'])
    assert m['golden_status'] == 'LOGIC_ERROR'
    assert m['verification_meta']['strategy_b']['flag_errors'] > 0
