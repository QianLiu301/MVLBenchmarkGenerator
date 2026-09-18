"""Homepage v3 acceptance tests (run: python -m pytest tests -q).

Uses an isolated SQLite database so they never touch Neon.
"""
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'web'))


@pytest.fixture(scope='module')
def app(tmp_path_factory):
    db = tmp_path_factory.mktemp('db') / 'test.db'
    os.environ['DATABASE_URL'] = f"sqlite:///{db.as_posix()}"
    os.environ['ENABLE_PROXY'] = 'false'
    import importlib
    import library.db as dbmod
    dbmod._engine = None
    dbmod._Session = None
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config['TESTING'] = True
    return appmod.app


@pytest.fixture(scope='module')
def client(app):
    return app.test_client()


def _seed(n_alu=3, n_reg=1):
    from library import service
    from library.db import session_scope
    with session_scope() as s:
        for i in range(n_alu):
            spec = service.describe_spec('alu', 3, 8 + 2 * i, ['ADD', 'SUB'])
            bm = service.get_or_create_benchmark(s, spec)
            service.add_implementation(s, bm, 'c', f'int main(void){{return {i};}}\n',
                                       metrics={'golden_status': 'PASS' if i % 2 == 0 else 'LOGIC_ERROR',
                                                'sim_status': 'pass'})
        for i in range(n_reg):
            spec = service.describe_spec('register', 3, 8, ['ADD'], {'register_count': 16})
            bm = service.get_or_create_benchmark(s, spec)
            service.add_implementation(s, bm, 'verilog', 'module m; endmodule\n',
                                       metrics={'golden_status': 'PASS', 'sim_status': 'pass'})


def test_empty_state_is_compact(client):
    html = client.get('/').get_data(as_text=True)
    assert 'No benchmarks are published yet' in html
    assert 'empty-row' in html
    # every chip is present but disabled at zero
    assert html.count('aria-disabled="true"') == 3


def test_home_counts_match_library(client):
    _seed()
    from library import service
    from library.db import session_scope
    with session_scope() as s:
        modules = {m['value']: m['specs'] for m in service.module_counts(s)}
        total = service.stats(s)['benchmarks']
        for value, n in modules.items():
            _, lib_total = service.list_benchmarks(s, {'module_type': value}, page=None)
            assert n == lib_total, f'homepage {value}={n} but library={lib_total}'
        _, all_total = service.list_benchmarks(s, {}, page=None)
        assert total == all_total
    html = client.get('/').get_data(as_text=True)
    m = re.search(r'All <span class="n">(\d+)</span>', html)
    assert m and int(m.group(1)) == total
    assert re.search(r'ALU <span class="n">3</span>', html)
    assert re.search(r'Register file <span class="n">1</span>', html)
    assert 'Processor <span class="n">0</span>' in html and 'aria-disabled="true"' in html


def test_bibtex_identical_everywhere(client):
    from library import service
    from library.db import session_scope
    file_text = service.citation_bibtex()
    home = client.get('/').get_data(as_text=True)
    cite = client.get('/cite').get_data(as_text=True)
    assert file_text.strip() in home and file_text.strip() in cite
    assert client.get('/citation.bib').get_data(as_text=True) == file_text
    with session_scope() as s:
        rows, _ = service.list_benchmarks(s, {}, page=None)
        data = service.build_zip(rows)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.read('CITATION.bib').decode('utf-8') == file_text
        manifest = json.loads(zf.read('manifest.json'))
        assert manifest['format_version'] == '1.0'
        assert all('sha256' in f for f in manifest['files'])
    assert 'url       = {https://llm-mvl.com}' in file_text
    assert 'note' not in file_text


def test_selection_counter_and_limit(client):
    r = client.get('/api/library/selection?module_type=alu&verified_only=1')
    d = r.get_json()
    assert d['specs'] == 2 and d['files'] == 2   # verified only: specs whose impl (0, 2) is PASS
    r = client.get('/library/download-selection?module_type=alu&verified_only=0')
    assert r.status_code == 200 and r.mimetype == 'application/zip'
    with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
        manifest = json.loads(zf.read('manifest.json'))
        assert len(manifest['files']) == 3 and manifest['filters'] == {'module_type': ['alu']}


def test_header_and_nav(client):
    html = client.get('/').get_data(as_text=True)
    assert 'brand/mvl-logo-horizontal.svg' in html
    assert 'aria-label="MVL Benchmark Library, home"' in html
    assert 'apple-touch-icon' in html
    nav = html[html.index('<nav class="site-nav"'):html.index('</nav>')]
    assert '>Cite<' not in nav
    assert client.get('/static/brand/favicon-32.png').status_code == 200
