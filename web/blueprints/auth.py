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

from flask import (Blueprint, flash, redirect, render_template, request,
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
        return url_for('generate_page')
    parsed = urlparse(target)
    if parsed.netloc or parsed.scheme or not target.startswith('/'):
        return url_for('generate_page')
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
