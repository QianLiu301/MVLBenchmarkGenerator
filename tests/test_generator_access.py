"""Reviewer access to the generator: the review password opens /generate and its API,
never the maintainer area, and never DeepSeek. Test passwords only; no LLM is called."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'web'))


@pytest.fixture
def app(monkeypatch):
    from app import app as flask_app
    import blueprints.auth as auth
    monkeypatch.setattr(auth, 'ACCESS_PASSWORD', 'test-maintainer')
    monkeypatch.setattr(auth, 'SITE_PASSWORD', 'test-review')
    monkeypatch.delenv('GENERATOR_PASSWORD', raising=False)
    flask_app.config['TESTING'] = True
    return flask_app


def test_site_gate_then_generator(app):
    c = app.test_client()
    assert c.get('/generate').status_code == 302                       # site gate first
    c.post('/site-login', data={'password': 'test-review', 'next': '/'})
    r = c.get('/generate')
    assert r.status_code == 200 and b'value="deepseek"' not in r.data
    assert c.get('/admin/').status_code == 302                         # not the maintainer area
    r = c.post('/api/generate', json={'llm': 'deepseek'})
    assert r.status_code == 403


def test_generator_password_without_site_gate(app, monkeypatch):
    """After the review the site gate may be removed; GENERATOR_PASSWORD then still opens /generate."""
    import blueprints.auth as auth
    monkeypatch.setattr(auth, 'SITE_PASSWORD', '')
    monkeypatch.setenv('GENERATOR_PASSWORD', 'test-gen')
    c = app.test_client()
    assert c.get('/generate').status_code == 302
    r = c.post('/login', data={'password': 'test-gen', 'next': '/generate'})
    assert r.status_code == 302
    assert c.get('/generate').status_code == 200
    assert c.get('/admin/').status_code == 302


def test_maintainer_keeps_everything(app):
    c = app.test_client()
    c.post('/site-login', data={'password': 'test-review', 'next': '/'})
    c.post('/login', data={'password': 'test-maintainer', 'next': '/'})
    assert c.get('/admin/').status_code == 200
    assert b'value="deepseek"' in c.get('/generate').data


def test_wrong_password(app):
    c = app.test_client()
    c.post('/login', data={'password': 'nope', 'next': '/generate'})
    r = c.post('/api/generate', json={'llm': 'gemini'})
    assert r.status_code in (302, 401)


def test_separate_generator_password(app, monkeypatch):
    """GENERATOR_PASSWORD set: the site password no longer opens the generator; the
    generator password opens the site and the generator, never the maintainer area."""
    monkeypatch.setenv('GENERATOR_PASSWORD', 'test-gen')
    lib = app.test_client()
    lib.post('/site-login', data={'password': 'test-review', 'next': '/'})
    assert lib.get('/library').status_code == 200
    assert lib.get('/generate').status_code == 302
    assert lib.post('/api/generate', json={'llm': 'gemini'}).status_code == 401

    gen = app.test_client()
    login = gen.get('/site-login?next=/generate').get_data(as_text=True)
    assert 'Library' not in login and 'library' not in login.replace('library.css', '')
    gen.post('/site-login', data={'password': 'test-gen', 'next': '/generate'})
    page = gen.get('/generate').get_data(as_text=True)
    assert 'href="/library"' not in page and 'class="brand"' not in page and 'Maintainer' not in page
    r = gen.get('/library')                                  # the library stays out of sight
    assert r.status_code == 302 and r.headers['Location'].endswith('/generate')
    assert gen.get('/api/v1/benchmarks').status_code == 404
    assert gen.get('/admin/').status_code == 302
    assert gen.post('/api/generate', json={'llm': 'deepseek'}).status_code == 403

    lib.post('/login', data={'password': 'test-gen', 'next': '/generate'})   # a library reviewer given both
    page = lib.get('/generate').get_data(as_text=True)
    assert 'href="/library"' in page                         # keeps the full navigation
