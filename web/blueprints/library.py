"""Public, read-only library pages: home, browse/filter, detail, code view, downloads."""
import io
import math
from datetime import datetime

from flask import (Blueprint, Response, abort, jsonify, redirect, render_template,
                   request, send_file, url_for)

from library import service
from library.db import session_scope
from library.models import (LANGUAGES, MODULE_ICONS, MODULE_TYPES, SOURCES)

bp = Blueprint('library', __name__)

_LABELS = {
    'module_labels': MODULE_TYPES,
    'module_icons': MODULE_ICONS,
    'language_labels': LANGUAGES,
    'source_labels': SOURCES,
    'family_labels': service.FAMILY_LABELS,
    'radix_names': service.RADIX_NAMES,
}

_FILTER_KEYS = list(service.FILTER_FIELDS) + list(service.IMPL_FILTERS) + ['q']


def _current_filters() -> dict:
    return {k: request.args.get(k) for k in _FILTER_KEYS if request.args.get(k)}


@bp.route('/')
def home():
    with session_scope() as s:
        modules = service.module_counts(s)
        only_alu = all(m['specs'] == 0 for m in modules if m['value'] != 'alu')
        return render_template('library/home.html', stats=service.stats(s),
                               modules=modules, only_alu=only_alu,
                               recent=service.recent_benchmarks(s, 5),
                               bibtex=service.bibtex(), **_LABELS)


@bp.route('/library')
def browse():
    current = _current_filters()
    sort = request.args.get('sort', 'name')
    if sort not in service.SORT_OPTIONS:
        sort = 'name'
    try:
        page = max(int(request.args.get('page', 1)), 1)
    except ValueError:
        page = 1
    with session_scope() as s:
        benchmarks, total = service.list_benchmarks(s, current, sort=sort, page=page)
        pages = max(math.ceil(total / service.PAGE_SIZE), 1)
        return render_template('library/browse.html', benchmarks=benchmarks, total=total,
                               page=page, pages=pages, sort=sort, current=current,
                               facets=service.facets(s), **_LABELS)


@bp.route('/benchmark/<slug>')
def detail(slug):
    with session_scope() as s:
        b = service.get_benchmark(s, slug)
        if b is None or b.status != 'published':
            abort(404)
        impls = b.published_implementations
        events = list(b.review_events)
        return render_template('library/detail.html', b=b, impls=impls,
                               verified_n=sum(1 for i in impls if i.golden_status == 'PASS'),
                               ops=service.op_definitions(b), events=events,
                               bibtex=service.bibtex(b), **_LABELS)


@bp.route('/library/<slug>')
def detail_legacy(slug):
    return redirect(url_for('library.detail', slug=slug), code=301)


@bp.route('/benchmark/<slug>/spec.json')
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


@bp.route('/benchmark/<slug>/<int:impl_id>')
@bp.route('/library/<slug>/<int:impl_id>')
def view_code(slug, impl_id):
    with session_scope() as s:
        b, i = _load_impl(s, slug, impl_id)
        return render_template('library/code.html', b=b, i=i, report=i.verification_report or {}, **_LABELS)


@bp.route('/benchmark/<slug>/<int:impl_id>/log')
def view_log(slug, impl_id):
    with session_scope() as s:
        b, i = _load_impl(s, slug, impl_id)
        return Response(i.verification_log or '(no log recorded)', mimetype='text/plain')


@bp.route('/benchmark/<slug>/<int:impl_id>/download')
@bp.route('/library/<slug>/<int:impl_id>/download')
def download_impl(slug, impl_id):
    with session_scope() as s:
        b, i = _load_impl(s, slug, impl_id)
        service.bump_downloads(s, benchmark=b, impl=i)
        data = i.code.encode('utf-8')
        name = i.filename
    return send_file(io.BytesIO(data), as_attachment=True, download_name=name, mimetype='text/plain')


@bp.route('/benchmark/<slug>/download')
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
        benchmarks, _ = service.list_benchmarks(s, current, sort='name', page=None)
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


@bp.route('/format')
def format():
    from library.submissions import MANIFEST_TEMPLATE
    import json
    return render_template('library/format.html', template_json=json.dumps(MANIFEST_TEMPLATE, indent=2))


@bp.route('/acknowledgements')
def acknowledgements():
    with session_scope() as s:
        return render_template('library/acknowledgements.html',
                               contributors=service.contributors(s), **_LABELS)


@bp.route('/api/library/stats')
def api_stats():
    with session_scope() as s:
        return jsonify(service.stats(s))
