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
