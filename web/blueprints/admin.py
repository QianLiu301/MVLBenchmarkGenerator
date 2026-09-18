"""Maintainer review queue (password-protected)."""
from flask import (Blueprint, abort, flash, redirect, render_template, request, url_for)

from blueprints.auth import require_access
from library import submissions
from library.db import session_scope
from library.models import LANGUAGES, News, PIPELINE_STEPS, Submission
from datetime import datetime
from sqlalchemy import select

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


# ---------------------------------------------------------------------------
# News (homepage "News" block)
# ---------------------------------------------------------------------------

@bp.route('/news')
@require_access
def news_list():
    with session_scope() as s:
        items = list(s.execute(select(News).order_by(News.date.desc())).scalars())
        return render_template('admin/news.html', items=items, item=None)


@bp.route('/news/new', methods=['GET', 'POST'])
@bp.route('/news/<int:news_id>', methods=['GET', 'POST'])
@require_access
def news_edit(news_id=None):
    with session_scope() as s:
        item = s.get(News, news_id) if news_id else None
        if news_id and item is None:
            abort(404)
        if request.method == 'POST':
            title = (request.form.get('title') or '').strip()
            if not title:
                flash('A title is required.', 'error')
                return redirect(request.path)
            if item is None:
                item = News()
                s.add(item)
            item.title = title
            item.body_md = request.form.get('body_md') or ''
            item.link = (request.form.get('link') or '').strip() or None
            item.status = 'published' if request.form.get('status') == 'published' else 'draft'
            try:
                item.date = datetime.fromisoformat(request.form.get('date') or '')
            except ValueError:
                item.date = item.date or datetime.utcnow()
            s.flush()
            flash('News item saved.', 'success')
            return redirect(url_for('admin.news_list'))
        items = list(s.execute(select(News).order_by(News.date.desc())).scalars())
        return render_template('admin/news.html', items=items, item=item)


@bp.route('/news/<int:news_id>/delete', methods=['POST'])
@require_access
def news_delete(news_id):
    with session_scope() as s:
        item = s.get(News, news_id)
        if item:
            s.delete(item)
            flash('News item deleted.', 'info')
    return redirect(url_for('admin.news_list'))
