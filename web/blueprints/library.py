"""Public, read-only library pages: home, browse/filter, detail, code view, downloads."""
import io
import math
from datetime import datetime

from flask import (Blueprint, Response, abort, jsonify, redirect, render_template,
                   request, send_file, url_for)

from sqlalchemy import select

from library import service
from library.db import session_scope
from library.models import (LANGUAGES, MODULE_ICONS, MODULE_TYPES, News, SOURCES)

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
    module = request.args.get('module') or None
    if module not in MODULE_TYPES:
        module = None
    with session_scope() as s:
        modules = service.module_counts(s)
        news = list(s.execute(select(News).where(News.status == 'published')
                              .order_by(News.date.desc()).limit(3)).scalars())
        return render_template('library/home.html', stats=service.stats(s),
                               modules=modules, module=module,
                               latest=service.latest_benchmarks(s, 10, module),
                               news=news, bibtex=service.bibtex(), **_LABELS)


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
    download_mode = request.args.get('mode') == 'download'
    with session_scope() as s:
        benchmarks, total = service.list_benchmarks(s, current, sort=sort, page=page)
        pages = max(math.ceil(total / service.PAGE_SIZE), 1)
        facets = service.facets(s)
        selection = service.normalize_selection(request.args) if download_mode else None
        summary = service.selection_summary(s, selection) if download_mode else None
        return render_template('library/browse.html', benchmarks=benchmarks, total=total,
                               page=page, pages=pages, sort=sort, current=current,
                               facets=facets, download_mode=download_mode,
                               selection=selection, summary=summary, **_LABELS)


@bp.route('/api/library/selection')
def api_selection():
    """Live counter for the download-by-selection panel."""
    sel = service.normalize_selection(request.args)
    with session_scope() as s:
        return jsonify(service.selection_summary(s, sel))


@bp.route('/library/download-selection')
def download_selection():
    sel = service.normalize_selection(request.args)
    with session_scope() as s:
        data, err = service.selection_zip(s, sel)
    if err:
        return render_template('library/error.html', message=err), 400
    stamp = datetime.utcnow().strftime('%Y%m%d')
    return send_file(io.BytesIO(data), as_attachment=True,
                     download_name=f"mvl-benchmarks-selection-{stamp}.zip", mimetype='application/zip')


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
    return render_template('library/cite.html', bibtex=service.bibtex(), citation_text=service.citation_text())


@bp.route('/citation.bib')
def citation_file():
    return Response(service.bibtex(), mimetype='application/x-bibtex',
                    headers={'Content-Disposition': 'attachment; filename=CITATION.bib'})


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
