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


# ----------------------------------------------------------------------------
# Reviewer access to the generator only (not to the maintainer area)
# ----------------------------------------------------------------------------
# Reviewers use the generator with the review password: GENERATOR_PASSWORD if it is
# set, otherwise the whole-site SITE_PASSWORD (defined below). With a separate
# GENERATOR_PASSWORD the two groups of reviewers are kept apart: the site password
# opens the library but not the generator; the generator password opens both (the
# generator page links to the library). Neither opens the review queue, and reviewer
# sessions cannot call DeepSeek, which runs on a privately paid account.
def review_password() -> str:
    return os.environ.get('GENERATOR_PASSWORD', '').strip() or SITE_PASSWORD


def can_generate() -> bool:
    if is_authed() or session.get('gen_ok'):
        return True
    # a reviewer who has entered the same password at the site gate is not asked again
    return bool(review_password()) and review_password() == SITE_PASSWORD and bool(session.get('site_ok'))


def generator_only() -> bool:
    """A reviewer of the generator paper: signed in with GENERATOR_PASSWORD only. The
    generator is then shown on its own, without the library around it."""
    return bool(session.get('gen_ok')) and not session.get('site_ok') and not is_authed()


# What a generator-only session may reach: the page, the API calls it makes, sign-out.
_GENERATOR_PATHS = ('/generate', '/logout', '/api/status', '/api/tools', '/api/check-tools',
                    '/api/generate', '/api/generate-stream', '/api/validate', '/api/validate-b',
                    '/api/validate-both', '/api/run-simulation', '/api/download-zip')


def require_generator(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not can_generate():
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Sign in to use the generator.'}), 401
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
        if review_password() and hmac.compare_digest(supplied, review_password()):
            session['gen_ok'] = True          # generator only
            session.permanent = True
            return redirect(next_url)
        flash('Incorrect password. Please try again.', 'error')
    if is_authed() or (can_generate() and next_url.startswith('/generate')):
        return redirect(next_url)
    if next_url.startswith('/generate'):
        return render_template('generator_login.html', next_url=next_url, action=url_for('auth.login'))
    return render_template('login.html', next_url=next_url)


@bp.route('/logout')
def logout():
    gen_only = generator_only()
    session.pop('authed', None)
    session.pop('gen_ok', None)
    session.pop('site_ok', None)      # signing out ends every login, the site password's too
    return redirect(url_for('generate_page') if gen_only else url_for('library.home'))


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
    if generator_only():                            # generator reviewers see the generator alone
        if path in _GENERATOR_PATHS:
            return None
        if path.startswith('/api/'):
            return jsonify({'error': 'Not available.'}), 404
        return redirect(url_for('generate_page'))
    if session.get('site_ok') or is_authed():      # the maintainer login opens the site too
        return None
    if path.startswith('/api/'):
        return jsonify({'error': 'This site is password-protected.'}), 401
    return redirect(url_for('auth.site_login', next=request.full_path.rstrip('?')))


@bp.route('/site-login', methods=['GET', 'POST'])
def site_login():
    next_url = _safe_next(request.values.get('next', ''))
    if not SITE_PASSWORD or session.get('site_ok') or session.get('gen_ok'):
        return redirect(next_url)
    if request.method == 'POST':
        supplied = request.form.get('password', '')
        if hmac.compare_digest(supplied, SITE_PASSWORD):
            session['site_ok'] = True
            session.permanent = True
            return redirect(next_url)
        if review_password() and hmac.compare_digest(supplied, review_password()):
            session['gen_ok'] = True          # generator reviewers: the site and the generator
            session.permanent = True
            return redirect(next_url)
        flash('Incorrect password. Please try again.', 'error')
    if next_url.startswith('/generate'):     # the generator's reviewers never see the library's name
        return render_template('generator_login.html', next_url=next_url, action=url_for('auth.site_login'))
    return render_template('site_login.html', next_url=next_url)
