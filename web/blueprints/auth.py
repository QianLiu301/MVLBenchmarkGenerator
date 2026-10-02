"""Server-side password gate for the generator (and later the admin area).

Replaces the previous client-side check, whose password was readable in the page
source and bypassable by setting sessionStorage. One shared password from the
ACCESS_PASSWORD environment variable; Flask's signed session cookie carries the
login state.
"""
import hmac
import os
from functools import wraps
from urllib.parse import urlparse

from flask import (Blueprint, flash, jsonify, redirect, render_template, request,
                   session, url_for)

bp = Blueprint('auth', __name__)

# Falls back to the historical password so an un-configured deployment keeps
# working; set ACCESS_PASSWORD on Render to change it without a commit.
ACCESS_PASSWORD = os.environ.get('ACCESS_PASSWORD', 'rolf2026')


def is_authed() -> bool:
    return bool(session.get('authed'))


def _safe_next(target: str) -> str:
    """Only allow redirects within this site."""
    if not target:
        return url_for('library.home')
    parsed = urlparse(target)
    if parsed.netloc or parsed.scheme or not target.startswith('/'):
        return url_for('library.home')
    return target


def require_access(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not is_authed():
            return redirect(url_for('auth.login', next=request.full_path.rstrip('?')))
        return view(*args, **kwargs)
    return wrapped


@bp.route('/login', methods=['GET', 'POST'])
def login():
    next_url = _safe_next(request.values.get('next', ''))
    if request.method == 'POST':
        supplied = request.form.get('password', '')
        if hmac.compare_digest(supplied, ACCESS_PASSWORD):
            session['authed'] = True
            session.permanent = True
            return redirect(next_url)
        flash('Incorrect password. Please try again.', 'error')
    if is_authed():
        return redirect(next_url)
    return render_template('login.html', next_url=next_url)


@bp.route('/logout')
def logout():
    session.pop('authed', None)
    return redirect(url_for('library.home'))


# ----------------------------------------------------------------------------
# Whole-site password (separate from ACCESS_PASSWORD, which guards the generator)
# ----------------------------------------------------------------------------
# Set SITE_PASSWORD on Render to close the whole site, e.g. while the paper is under
# review and only reviewers should see it; remove the variable to open it again.
# Unset (the default) means the site is public.
SITE_PASSWORD = os.environ.get('SITE_PASSWORD', '').strip()

# Reachable without the site password: the login page itself, the health check that
# Docker and Render poll, robots.txt and the static assets the login page needs.
_SITE_OPEN = ('/site-login', '/api/status', '/robots.txt', '/favicon.ico')


@bp.before_app_request
def site_gate():
    if not SITE_PASSWORD:
        return None
    path = request.path
    if path in _SITE_OPEN or path.startswith('/static/'):
        return None
    if session.get('site_ok') or is_authed():      # the maintainer login opens the site too
        return None
    if path.startswith('/api/'):
        return jsonify({'error': 'This site is password-protected.'}), 401
    return redirect(url_for('auth.site_login', next=request.full_path.rstrip('?')))


@bp.route('/site-login', methods=['GET', 'POST'])
def site_login():
    next_url = _safe_next(request.values.get('next', ''))
    if not SITE_PASSWORD or session.get('site_ok'):
        return redirect(next_url)
    if request.method == 'POST':
        if hmac.compare_digest(request.form.get('password', ''), SITE_PASSWORD):
            session['site_ok'] = True
            session.permanent = True
            return redirect(next_url)
        flash('Incorrect password. Please try again.', 'error')
    return render_template('site_login.html', next_url=next_url)
