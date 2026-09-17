"""Maintainer review queue (password-protected)."""
from flask import (Blueprint, abort, flash, redirect, render_template, request, url_for)

from blueprints.auth import require_access
from library import submissions
from library.db import session_scope
from library.models import LANGUAGES, PIPELINE_STEPS, Submission

bp = Blueprint('admin', __name__, url_prefix='/admin')


@bp.route('/')
@require_access
def queue():
    status = request.args.get('status', 'awaiting_review')
    with session_scope() as s:
        subs = submissions.list_submissions(s, None if status == 'all' else status)
        counts = {st: len(submissions.list_submissions(s, st))
                  for st in ('awaiting_review', 'running', 'queued', 'failed', 'approved', 'rejected')}
        return render_template('admin/queue.html', subs=subs, status=status, counts=counts,
                               steps=PIPELINE_STEPS)


@bp.route('/<int:sub_id>')
@require_access
def review(sub_id):
    with session_scope() as s:
        sub = s.get(Submission, sub_id)
        if sub is None:
            abort(404)
        return render_template('admin/review.html', sub=sub, steps=PIPELINE_STEPS,
                               language_labels=LANGUAGES)


@bp.route('/<int:sub_id>/decide', methods=['POST'])
@require_access
def decide(sub_id):
    action = request.form.get('action')
    reason = (request.form.get('reason') or '').strip()
    if len(reason) < 5:
        flash('A reason (at least 5 characters) is required for every decision.', 'error')
        return redirect(url_for('admin.review', sub_id=sub_id))
    try:
        if action == 'approve':
            slug = submissions.approve(sub_id, reason)
            flash(f'Approved and published under {slug}.', 'success')
            return redirect(url_for('library.detail', slug=slug))
        elif action == 'reject':
            submissions.reject(sub_id, reason)
            flash('Submission rejected.', 'info')
        else:
            flash('Unknown action.', 'error')
    except ValueError as e:
        flash(str(e), 'error')
    return redirect(url_for('admin.queue'))
