"""Public submission: form, manifest template/schema, status page with the pipeline steps."""
import json
from pathlib import Path

from flask import (Blueprint, Response, abort, jsonify, render_template, request)

from library import submissions
from library.db import session_scope
from library.models import LANGUAGE_EXT, LANGUAGES, MODULE_TYPES, PIPELINE_STEPS_SHOWN

bp = Blueprint('submit', __name__)

# every language the library can verify; the form and the schema read the same table
_ALLOWED_EXT = tuple(LANGUAGE_EXT.values())


@bp.route('/submit')
def form():
    return render_template('submit/form.html', module_labels=MODULE_TYPES,
                           language_labels=LANGUAGES, language_ext=LANGUAGE_EXT, steps=PIPELINE_STEPS_SHOWN,
                           schema_json=json.dumps(submissions.MANIFEST_SCHEMA),
                           template_json=json.dumps(submissions.MANIFEST_TEMPLATE, indent=2),
                           example=_example())


# A complete submission that passes every check (tests/test_submit_example.py keeps it so)
_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / 'static' / 'examples' / 'alu_k3_8t'


def _example():
    manifest = (_EXAMPLE_DIR / 'manifest.json').read_text(encoding='utf-8')
    code_name = json.loads(manifest)['files'][0]['filename']
    return {'manifest': manifest, 'code_name': code_name,
            'code': (_EXAMPLE_DIR / code_name).read_text(encoding='utf-8')}


@bp.route('/submit/manifest-template.json')
def manifest_template():
    return Response(json.dumps(submissions.MANIFEST_TEMPLATE, indent=2), mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=manifest.json'})


@bp.route('/submit/manifest-schema.json')
def manifest_schema():
    return Response(json.dumps(submissions.MANIFEST_SCHEMA, indent=2), mimetype='application/json')


@bp.route('/api/submit', methods=['POST'])
def api_submit():
    """multipart/form-data: manifest (file or field 'manifest') + code files."""
    manifest_raw = None
    files = {}
    for f in request.files.getlist('files'):
        name = (f.filename or '').rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        if not name:
            continue
        data = f.read()
        if name.lower() == 'manifest.json':
            manifest_raw = data
            continue
        if not name.lower().endswith(_ALLOWED_EXT):
            return jsonify({'ok': False, 'errors': [f"{name}: only {' '.join(_ALLOWED_EXT)} files (and manifest.json) are accepted"]}), 400
        if len(data) > submissions.MAX_FILE_BYTES:
            return jsonify({'ok': False, 'errors': [f"{name}: larger than {submissions.MAX_FILE_BYTES // 1024} KB"]}), 400
        try:
            files[name] = data.decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'ok': False, 'errors': [f"{name}: not UTF-8 text"]}), 400
    if manifest_raw is None and request.form.get('manifest'):
        manifest_raw = request.form['manifest'].encode('utf-8')
    if manifest_raw is None:
        return jsonify({'ok': False, 'errors': ['manifest.json is missing']}), 400
    try:
        manifest = json.loads(manifest_raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as e:
        return jsonify({'ok': False, 'errors': [f'manifest.json is not valid JSON: {e}']}), 400
    if len(files) > submissions.MAX_FILES:
        return jsonify({'ok': False, 'errors': [f'at most {submissions.MAX_FILES} files']}), 400

    token, errs = submissions.create_submission(manifest, files)
    return jsonify({'ok': not errs, 'token': token, 'errors': errs,
                    'status_url': f'/submission/{token}'}), (200 if not errs else 422)


@bp.route('/submission/<token>')
def status(token):
    with session_scope() as s:
        sub = submissions.get_by_token(s, token)
        if sub is None:
            abort(404)
        return render_template('submit/status.html', sub=sub, steps=PIPELINE_STEPS_SHOWN,
                               language_labels=LANGUAGES, outcome=submissions.outcome(sub))


@bp.route('/api/submission/<token>')
def api_status(token):
    with session_scope() as s:
        sub = submissions.get_by_token(s, token)
        if sub is None:
            abort(404)
        return jsonify({'status': sub.status, 'steps': sub.steps, 'slug': sub.slug,
                        'results': sub.results, 'decision_reason': sub.decision_reason})
