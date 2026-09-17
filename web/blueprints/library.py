"""Public, read-only library pages: home, browse/filter, detail, code view, downloads."""
import io
from datetime import datetime

from flask import (Blueprint, Response, abort, jsonify, render_template,
                   request, send_file)

from library import service
from library.db import session_scope
from library.models import LANGUAGES, MODULE_TYPES, SOURCES

bp = Blueprint('library', __name__)

_LABELS = {
    'module_labels': MODULE_TYPES,
    'language_labels': LANGUAGES,
    'source_labels': SOURCES,
    'family_labels': service.FAMILY_LABELS,
    'radix_names': service.RADIX_NAMES,
}


def _current_filters() -> dict:
    """Filter values from the query string, only the ones we understand."""
    keys = list(service.FILTER_FIELDS) + ['language', 'source', 'verified']
    return {k: request.args.get(k) for k in keys if request.args.get(k)}


@bp.route('/')
def home():
    with session_scope() as s:
        return render_template('library/home.html', stats=service.stats(s),
                               facets=service.facets(s), **_LABELS)


@bp.route('/library')
def browse():
    current = _current_filters()
    with session_scope() as s:
        benchmarks = service.list_benchmarks(
            s, current, language=current.get('language'), source=current.get('source'),
            verified_only=bool(current.get('verified')))
        # touch relationships while the session is open
        for b in benchmarks:
            _ = b.published_implementations
        return render_template('library/browse.html', benchmarks=benchmarks, current=current,
                               facets=service.facets(s), **_LABELS)


@bp.route('/library/<slug>')
def detail(slug):
    with session_scope() as s:
        b = service.get_benchmark(s, slug)
        if b is None or b.status != 'published':
            abort(404)
        impls = b.published_implementations
        verified_n = sum(1 for i in impls if i.golden_status == 'PASS')
        return render_template('library/detail.html', b=b, impls=impls, verified_n=verified_n,
                               bibtex=service.bibtex(b), **_LABELS)


@bp.route('/library/<slug>/spec.json')
def spec_json(slug):
    with session_scope() as s:
        b = service.get_benchmark(s, slug)
        if b is None:
            abort(404)
        return Response(service.spec_json(b), mimetype='application/json')


def _load_impl(s, slug, impl_id):
    b = service.get_benchmark(s, slug)
    if b is None:
        abort(404)
    impl = next((i for i in b.implementations if i.id == impl_id and i.status == 'published'), None)
    if impl is None:
        abort(404)
    return b, impl


@bp.route('/library/<slug>/<int:impl_id>')
def view_code(slug, impl_id):
    with session_scope() as s:
        b, i = _load_impl(s, slug, impl_id)
        return render_template('library/code.html', b=b, i=i, report=i.verification_report or {}, **_LABELS)


@bp.route('/library/<slug>/<int:impl_id>/download')
def download_impl(slug, impl_id):
    with session_scope() as s:
        b, i = _load_impl(s, slug, impl_id)
        service.bump_downloads(s, benchmark=b, impl=i)
        data = i.code.encode('utf-8')
        name = i.filename
    return send_file(io.BytesIO(data), as_attachment=True, download_name=name, mimetype='text/plain')


@bp.route('/library/<slug>/download')
def download_benchmark(slug):
    with session_scope() as s:
        b = service.get_benchmark(s, slug)
        if b is None:
            abort(404)
        service.bump_downloads(s, benchmark=b)
        data = service.build_zip([b])
    return send_file(io.BytesIO(data), as_attachment=True, download_name=f"{slug}.zip",
                     mimetype='application/zip')


@bp.route('/library/download-all')
def download_all():
    current = _current_filters()
    with session_scope() as s:
        benchmarks = service.list_benchmarks(
            s, current, language=current.get('language'), source=current.get('source'),
            verified_only=bool(current.get('verified')))
        data = service.build_zip(benchmarks)
    stamp = datetime.utcnow().strftime('%Y%m%d')
    name = f"mvl-benchmarks-{stamp}.zip" if not current else f"mvl-benchmarks-selection-{stamp}.zip"
    return send_file(io.BytesIO(data), as_attachment=True, download_name=name, mimetype='application/zip')


@bp.route('/about')
def about():
    return render_template('library/about.html')


@bp.route('/cite')
def cite():
    return render_template('library/cite.html', bibtex=service.bibtex())


@bp.route('/api/library/stats')
def api_stats():
    with session_scope() as s:
        return jsonify(service.stats(s))
