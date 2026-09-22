"""Public JSON API and the pages that document it (run: python -m pytest tests -q).

Uses an isolated SQLite database so these never touch Neon.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'web'))


@pytest.fixture(scope='module')
def app(tmp_path_factory):
    db = tmp_path_factory.mktemp('db') / 'api.db'
    os.environ['DATABASE_URL'] = f"sqlite:///{db.as_posix()}"
    os.environ['ENABLE_PROXY'] = 'false'
    import importlib
    import library.db as dbmod
    dbmod._engine = None
    dbmod._Session = None
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config['TESTING'] = True
    _seed()
    return appmod.app


@pytest.fixture(scope='module')
def client(app):
    return app.test_client()


def _seed():
    """Two specs: k=3 (modular) with a verified C and a failed VHDL, k=4 (field) with a verified VHDL."""
    from library import service
    from library.db import session_scope
    with session_scope() as s:
        spec = service.describe_spec('alu', 3, 8, ['ADD', 'SUB'])
        bm = service.get_or_create_benchmark(s, spec)
        service.add_implementation(s, bm, 'c', 'int main(void){return 0;}\n',
                                   source='human-authored',
                                   metrics={'golden_status': 'PASS', 'sim_status': 'pass',
                                            'golden_passed': 42, 'golden_compared': 42,
                                            'verification_strength': 'random(N=50, seed=42)',
                                            'verification_meta': {'golden_model': '1.1'}})
        service.add_implementation(s, bm, 'vhdl', 'entity e is end entity;\n',
                                   provider='mistral', model_responded='codestral-latest',
                                   metrics={'golden_status': 'RUNTIME_ERROR', 'sim_status': 'fail',
                                            'golden_passed': 0, 'golden_compared': 0,
                                            'verification_strength': 'self-reported(N=0)'})
        spec = service.describe_spec('alu', 4, 8, ['ADD', 'MUL'])
        bm = service.get_or_create_benchmark(s, spec)
        service.add_implementation(s, bm, 'vhdl', 'entity f is end entity;\n',
                                   provider='gemini', model_responded='gemini-2.5-flash',
                                   metrics={'golden_status': 'PASS', 'sim_status': 'pass',
                                            'golden_passed': 137, 'golden_compared': 137,
                                            'verification_strength': 'random(N=50, seed=42)',
                                            'verification_meta': {'golden_model': '1.1'}})


def test_list_shape_and_paging(client):
    d = client.get('/api/v1/benchmarks').get_json()
    assert d['api_version'] == 1 and d['license'] == 'CC-BY-4.0'
    assert d['total'] == 2 and d['count'] == 2
    slugs = {b['slug'] for b in d['benchmarks']}
    assert slugs == {'alu_k3_8t', 'alu_k4_8t'}
    b = next(x for x in d['benchmarks'] if x['slug'] == 'alu_k3_8t')
    assert b['structure'] == 'Z/3⁸Z' and b['logic_family'] == 'modular'
    assert b['implementation_count'] == 2 and b['verified_count'] == 1
    assert b['languages'] == ['c', 'vhdl']
    assert 'implementations' not in b          # only with full=1
    assert b['url'].endswith('/benchmark/alu_k3_8t')

    one = client.get('/api/v1/benchmarks?limit=1&offset=1').get_json()
    assert one['total'] == 2 and one['count'] == 1 and one['offset'] == 1
    assert one['benchmarks'][0]['slug'] != d['benchmarks'][0]['slug']


def test_filters(client):
    assert client.get('/api/v1/benchmarks?k_value=4').get_json()['count'] == 1
    assert client.get('/api/v1/benchmarks?logic_family=field').get_json()['benchmarks'][0]['slug'] == 'alu_k4_8t'
    assert client.get('/api/v1/benchmarks?language=c').get_json()['count'] == 1
    assert client.get('/api/v1/benchmarks?k_value=99').get_json()['count'] == 0
    # both verification states and the per-model filter
    assert client.get('/api/v1/benchmarks?verified=1').get_json()['count'] == 2
    failed = client.get('/api/v1/benchmarks?verified=0').get_json()
    assert failed['count'] == 1 and failed['benchmarks'][0]['slug'] == 'alu_k3_8t'
    assert client.get('/api/v1/benchmarks?model=codestral-latest').get_json()['count'] == 1
    assert client.get('/api/v1/benchmarks?model=nope').get_json()['count'] == 0


def test_search_shorthands(client):
    """What a visitor types into the search box acts as a filter, not as text."""
    from library import service
    assert service.parse_query('k=3 vhdl')[0] == {'k_value': '3', 'language': 'vhdl'}
    assert service.parse_query('ternary')[0] == {'k_value': '3'}
    assert service.parse_query('GF(4)')[0] == {'logic_family': 'field'}
    assert service.parse_query('8 digits verified')[0] == {'bitwidth': '8', 'verified': '1'}
    assert service.parse_query('vhdl failed')[0] == {'language': 'vhdl', 'verified': '0'}
    # a specification name is a literal, not something to take apart
    assert service.parse_query('alu_k3_8t') == ({}, 'alu_k3_8t')
    # free text that means nothing to the parser survives as text
    assert service.parse_query('adder tree') == ({}, 'adder tree')

    got = client.get('/api/v1/benchmarks?q=k%3D3').get_json()
    assert got['count'] == 1 and got['benchmarks'][0]['slug'] == 'alu_k3_8t'
    assert client.get('/api/v1/benchmarks?q=GF(4)').get_json()['benchmarks'][0]['slug'] == 'alu_k4_8t'
    assert client.get('/api/v1/benchmarks?q=nonsense').get_json()['count'] == 0
    # an explicit filter wins over the shorthand
    assert client.get('/api/v1/benchmarks?q=k%3D3&k_value=4').get_json()['benchmarks'][0]['slug'] == 'alu_k4_8t'

    page = client.get('/library?q=k%3D3+vhdl').get_data(as_text=True)
    assert 'Search read as' in page and 'k = 3' in page and 'VHDL' in page


def test_full_and_detail_carry_verification(client):
    d = client.get('/api/v1/benchmarks/alu_k3_8t').get_json()
    impls = {i['language']: i for i in d['implementations']}
    assert impls['c']['verified'] is True and impls['c']['golden_model'] == 'PASS'
    assert impls['c']['vectors_compared'] == 42
    assert impls['c']['golden_model_version'] == '1.1'
    assert impls['vhdl']['verified'] is False and impls['vhdl']['golden_model'] == 'RUNTIME_ERROR'
    assert impls['vhdl']['model'] == 'codestral-latest'
    assert impls['c']['sha256'] and impls['c']['download_url'].endswith('/download')

    full = client.get('/api/v1/benchmarks?full=1').get_json()
    assert all('implementations' in b for b in full['benchmarks'])


def test_unknown_slug_is_404_json(client):
    r = client.get('/api/v1/benchmarks/does_not_exist')
    assert r.status_code == 404 and r.get_json()['error'] == 'not found'


def test_stats(client):
    d = client.get('/api/v1/stats').get_json()
    assert d['benchmarks'] == 2 and d['implementations'] == 3 and d['verified'] == 2
    assert d['format_version'] and d['golden_model_version']
    assert any(m['label'] == 'codestral-latest' for m in d['models'])
    assert 'k_value' in d['facets']


def test_statistics(client):
    """Coverage, rates and the gap list come from the same rows as the library itself."""
    from library import service
    from library.db import session_scope
    with session_scope() as s:
        d = service.library_statistics(s)
    assert d['ks'] == [3, 4] and d['digits'] == [8]
    # alu_k3_8t: verified C, failed VHDL -> 1 of 4 languages verified
    cell = d['matrix'][0]['cells'][0]
    assert cell['slug'] == 'alu_k3_8t' and cell['verified'] == 1 and cell['total'] == 4
    langs = {l['code']: l for l in d['languages']}
    assert langs['c']['verified'] == 1 and langs['c']['pct'] == 100
    assert langs['vhdl']['total'] == 2 and langs['vhdl']['verified'] == 1
    assert d['pairs_total'] == 8 and d['pairs_verified'] == 2
    # every pair without a passing implementation is listed, with what was tried
    gaps = {g['slug']: g for g in d['gaps']}
    assert 'VHDL' in gaps['alu_k3_8t']['missing']
    assert 'VHDL' in gaps['alu_k3_8t']['has_attempt']      # tried, did not pass
    assert 'C' in gaps['alu_k4_8t']['missing'] and not gaps['alu_k4_8t']['has_attempt']
    assert sum(d['strength'].values()) == 2                # both verified files

    page = client.get('/statistics').get_data(as_text=True)
    assert 'Coverage' in page and 'Where the collection is thin' in page
    assert client.get('/docs').get_data(as_text=True).count('Statistics') >= 1


def test_docs_pages_render(client):
    api = client.get('/api').get_data(as_text=True)
    assert '/api/v1/benchmarks' in api and 'CC BY 4.0' in api
    assert 'alu_k3_8t' in api                     # example rendered from real data
    started = client.get('/getting-started').get_data(as_text=True)
    assert 'ghdl -a --std=08' in started and 'alu_exec' in started
    home = client.get('/').get_data(as_text=True)
    assert 'What is the MVL Benchmark Library?' in home      # identity panel
    assert 'All specifications' in home and 'All implementations' in home   # category panels
    assert 'Getting started' in home and 'How it is verified' in home       # path cards
    docs = client.get('/docs').get_data(as_text=True)
    assert 'Benchmark format' in docs and 'Review process' in docs
