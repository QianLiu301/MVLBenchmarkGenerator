"""Submission pipeline: manifest schema → lint → simulation vs golden → formal →
LLM review → dedup → maintainer approval.

A submission is a manifest.json plus one code file per language. Steps 1–6 run
automatically in a background thread right after upload; the last step is a
human decision in /admin. Every step records status + log so the submitter's
status page and the admin queue show the same evidence.
"""
import json
import re
import secrets
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from . import service
from .db import session_scope
from .models import (LANGUAGE_EXT, LANGUAGES, MODULE_TYPES, PIPELINE_STEPS,
                     Submission, Implementation)

MAX_FILE_BYTES = 512 * 1024
MAX_FILES = 8
OPERATIONS = ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC']

# JSON Schema (draft-07 subset) — also served to the browser for pre-validation.
MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MVL Benchmark Library submission manifest",
    "type": "object",
    "required": ["manifest_version", "module_type", "k_value", "bitwidth", "operations",
                 "files", "submitter", "license"],
    "additionalProperties": False,
    "properties": {
        "manifest_version": {"const": 1},
        "module_type": {"type": "string", "enum": list(MODULE_TYPES.keys())},
        "k_value": {"type": "integer", "minimum": 2, "maximum": 16},
        "bitwidth": {"type": "integer", "minimum": 1, "maximum": 64},
        "operations": {"type": "array", "minItems": 1, "uniqueItems": True,
                       "items": {"type": "string", "enum": OPERATIONS}},
        "params": {"type": "object", "additionalProperties": False,
                   "properties": {"register_count": {"type": "integer", "minimum": 1},
                                  "pipeline_stages": {"type": "integer", "minimum": 1}}},
        "description": {"type": "string", "maxLength": 2000},
        "files": {"type": "array", "minItems": 1, "maxItems": MAX_FILES,
                  "items": {"type": "object", "required": ["filename", "language"],
                            "additionalProperties": False,
                            "properties": {"filename": {"type": "string", "pattern": r"^[A-Za-z0-9_.-]+\.(c|py|v|vhd)$"},
                                           "language": {"type": "string", "enum": list(LANGUAGES.keys())},
                                           "notes": {"type": "string", "maxLength": 1000}}}},
        "source": {"type": "string", "enum": ["human-authored", "llm-generated", "reference"]},
        "generator": {"type": "object", "additionalProperties": False,
                      "properties": {"provider": {"type": "string"}, "model": {"type": "string"},
                                     "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}},
        "submitter": {"type": "object", "required": ["name", "email"], "additionalProperties": False,
                      "properties": {"name": {"type": "string", "minLength": 2, "maxLength": 120},
                                     "email": {"type": "string", "format": "email", "maxLength": 160},
                                     "affiliation": {"type": "string", "maxLength": 200}}},
        "license": {"const": "CC-BY-4.0"},
    },
}

MANIFEST_TEMPLATE = {
    "manifest_version": 1,
    "module_type": "alu",
    "k_value": 3,
    "bitwidth": 8,
    "operations": ["ADD", "SUB", "MUL", "NEG", "INC", "DEC"],
    "params": {},
    "description": "Optional free-text description of the design.",
    "files": [
        {"filename": "alu_k3_8t.c", "language": "c", "notes": ""},
        {"filename": "alu_k3_8t.v", "language": "verilog", "notes": ""}
    ],
    "source": "human-authored",
    "generator": {"provider": "", "model": "", "prompt_sha256": ""},
    "submitter": {"name": "Your Name", "email": "you@example.org", "affiliation": "Your institution"},
    "license": "CC-BY-4.0"
}


# ----------------------------------------------------------------------------
# Step 1: schema (a small draft-07 validator — enough for MANIFEST_SCHEMA)
# ----------------------------------------------------------------------------

def _validate(schema: Dict, value, path: str = '$') -> List[str]:
    errs = []
    if 'const' in schema and value != schema['const']:
        return [f"{path}: must be {json.dumps(schema['const'])}"]
    if 'enum' in schema and value not in schema['enum']:
        return [f"{path}: must be one of {schema['enum']}"]
    t = schema.get('type')
    if t == 'object':
        if not isinstance(value, dict):
            return [f"{path}: must be an object"]
        for req in schema.get('required', []):
            if req not in value:
                errs.append(f"{path}.{req}: required")
        props = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            for key in value:
                if key not in props:
                    errs.append(f"{path}.{key}: unknown field")
        for key, sub in props.items():
            if key in value:
                errs += _validate(sub, value[key], f"{path}.{key}")
    elif t == 'array':
        if not isinstance(value, list):
            return [f"{path}: must be an array"]
        if 'minItems' in schema and len(value) < schema['minItems']:
            errs.append(f"{path}: at least {schema['minItems']} item(s)")
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            errs.append(f"{path}: at most {schema['maxItems']} item(s)")
        if schema.get('uniqueItems') and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            errs.append(f"{path}: items must be unique")
        for i, item in enumerate(value):
            errs += _validate(schema.get('items', {}), item, f"{path}[{i}]")
    elif t == 'integer':
        if not isinstance(value, int) or isinstance(value, bool):
            return [f"{path}: must be an integer"]
        if 'minimum' in schema and value < schema['minimum']:
            errs.append(f"{path}: must be ≥ {schema['minimum']}")
        if 'maximum' in schema and value > schema['maximum']:
            errs.append(f"{path}: must be ≤ {schema['maximum']}")
    elif t == 'string':
        if not isinstance(value, str):
            return [f"{path}: must be a string"]
        if 'minLength' in schema and len(value) < schema['minLength']:
            errs.append(f"{path}: at least {schema['minLength']} characters")
        if 'maxLength' in schema and len(value) > schema['maxLength']:
            errs.append(f"{path}: at most {schema['maxLength']} characters")
        if 'pattern' in schema and not re.match(schema['pattern'], value):
            errs.append(f"{path}: does not match {schema['pattern']}")
        if schema.get('format') == 'email' and not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', value):
            errs.append(f"{path}: must be an e-mail address")
    return errs


def validate_manifest(manifest: Dict, files: Dict[str, str]) -> List[str]:
    errs = _validate(MANIFEST_SCHEMA, manifest)
    if errs:
        return errs
    declared = {f['filename'] for f in manifest['files']}
    for name in declared:
        if name not in files:
            errs.append(f"file '{name}' is declared in the manifest but was not uploaded")
    for name in files:
        if name not in declared:
            errs.append(f"uploaded file '{name}' is not declared in manifest.files")
    for f in manifest['files']:
        ext = LANGUAGE_EXT[f['language']]
        if not f['filename'].endswith(ext):
            errs.append(f"file '{f['filename']}': language {f['language']} expects extension {ext}")
    if manifest.get('source') == 'llm-generated' and not (manifest.get('generator') or {}).get('model'):
        errs.append("generator.model is required when source is llm-generated")
    return errs


# ----------------------------------------------------------------------------
# Step 2: lint
# ----------------------------------------------------------------------------

_FORBIDDEN = {
    'c': [r'\bsystem\s*\(', r'\bfork\s*\(', r'\bexec[lv]p?\s*\(', r'#include\s*<windows\.h>', r'\bfopen\s*\('],
    'python': [r'\bimport\s+os\b', r'\bimport\s+subprocess\b', r'\bimport\s+socket\b', r'\bopen\s*\(', r'\b__import__\b', r'\beval\s*\(', r'\bexec\s*\('],
    'verilog': [r'\$system\b', r'\$fopen\b', r'\$fwrite\b'],
    'vhdl': [r'\bfile_open\b', r'\btextio\b.*\bfile\b'],
}
_REQUIRED = {
    'c': (r'\bint\s+main\s*\(', 'a main() function'),
    'python': (r'__main__', 'an if __name__ == "__main__" block'),
    'verilog': (r'\bmodule\s+\w+', 'a module declaration'),
    'vhdl': (r'\bentity\s+\w+\s+is\b', 'an entity declaration'),
}


def lint_file(name: str, code: str, language: str) -> List[str]:
    errs = []
    if len(code.encode('utf-8')) > MAX_FILE_BYTES:
        errs.append(f"{name}: larger than {MAX_FILE_BYTES // 1024} KB")
    if '\x00' in code:
        errs.append(f"{name}: binary content")
    if re.search(r'[^\x09\x0a\x0d\x20-\x7e -￿]', code):
        errs.append(f"{name}: control characters")
    req, what = _REQUIRED[language]
    if not re.search(req, code):
        errs.append(f"{name}: expected {what}")
    for pat in _FORBIDDEN.get(language, []):
        if re.search(pat, code):
            errs.append(f"{name}: forbidden construct matches /{pat}/")
    return errs


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def _set_step(sub: Submission, step: str, status: str, log: str = '', detail=None):
    steps = dict(sub.steps or {})
    steps[step] = {'status': status, 'log': (log or '')[-8000:], 'detail': detail,
                   'finished_at': datetime.utcnow().isoformat(timespec='seconds')}
    sub.steps = steps


def create_submission(manifest: Dict, files: Dict[str, str]) -> Tuple[Optional[str], List[str]]:
    """Validate schema (step 1) synchronously, store, and start the async steps."""
    errs = validate_manifest(manifest, files)
    with session_scope() as s:
        sub = Submission(
            token=secrets.token_urlsafe(12), manifest=manifest, files=files,
            slug=service.make_slug(manifest.get('module_type', 'alu'), manifest.get('k_value', 0),
                                   manifest.get('bitwidth', 0), manifest.get('params')),
            submitter_name=(manifest.get('submitter') or {}).get('name'),
            submitter_email=(manifest.get('submitter') or {}).get('email'),
            submitter_affiliation=(manifest.get('submitter') or {}).get('affiliation'),
            steps={step: {'status': 'pending'} for step, _ in PIPELINE_STEPS},
        )
        if errs:
            _set_step(sub, 'schema', 'fail', '\n'.join(errs))
            sub.status = 'failed'
        else:
            _set_step(sub, 'schema', 'pass', 'manifest and file list valid')
            sub.status = 'queued'
        s.add(sub)
        s.flush()
        token, sub_id = sub.token, sub.id
    if not errs:
        threading.Thread(target=run_pipeline, args=(sub_id,), daemon=True).start()
    return token, errs


def run_pipeline(sub_id: int):
    """Steps 2–6. Runs in a background thread; each step is persisted as it finishes."""
    try:
        from benchmark_validator import BenchmarkValidator
    except ImportError:
        from src.benchmark_validator import BenchmarkValidator
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        validator = BenchmarkValidator(project_root=str(service.PROJECT_ROOT))

    with session_scope() as s:
        sub = s.get(Submission, sub_id)
        sub.status = 'running'
        manifest, files = sub.manifest, sub.files

    def update(fn):
        with session_scope() as s:
            sub = s.get(Submission, sub_id)
            fn(sub)

    try:
        # --- 2. lint --------------------------------------------------------
        lint_errs = []
        for f in manifest['files']:
            lint_errs += lint_file(f['filename'], files[f['filename']], f['language'])
        if lint_errs:
            update(lambda sub: (_set_step(sub, 'lint', 'fail', '\n'.join(lint_errs)), setattr(sub, 'status', 'failed')))
            _skip_rest(sub_id, after='lint')
            return
        update(lambda sub: _set_step(sub, 'lint', 'pass', f"{len(files)} file(s) linted"))

        # --- 3. simulation vs golden ---------------------------------------
        update(lambda sub: _set_step(sub, 'simulation', 'running'))
        results = {}
        logs = []
        all_ok = True
        for f in manifest['files']:
            with contextlib.redirect_stdout(io.StringIO()):
                m = service.verify_code(files[f['filename']], f['language'], manifest['k_value'],
                                        manifest['bitwidth'], validator)
            results[f['filename']] = {k: v for k, v in m.items() if k not in ('verification_log',)}
            results[f['filename']]['verification_log'] = m['verification_log'][-4000:]
            results[f['filename']]['verified_at'] = m['verified_at'].isoformat(timespec='seconds')
            ok = m['golden_status'] == 'PASS'
            all_ok &= ok
            logs.append(f"{f['filename']}: sim={m['sim_status']} golden={m['golden_status']} "
                        f"{m['golden_passed']}/{m['golden_compared']} [{m['verification_strength']}]")
        update(lambda sub: (setattr(sub, 'results', results),
                            _set_step(sub, 'simulation', 'pass' if all_ok else 'fail', '\n'.join(logs))))
        if not all_ok:
            update(lambda sub: setattr(sub, 'status', 'failed'))
            _skip_rest(sub_id, after='simulation')
            return

        # --- 4. formal (optional) — no bounded-model checker wired yet --------
        update(lambda sub: _set_step(sub, 'formal', 'skipped', 'no formal back-end configured'))

        # --- 5. LLM review — pluggable; disabled until a reviewer is chosen ---
        update(lambda sub: _set_step(sub, 'llm_review', 'skipped', 'LLM reviewer not enabled'))

        # --- 6. dedup --------------------------------------------------------
        dups = []
        with session_scope() as s:
            for f in manifest['files']:
                d = service.find_duplicate(s, files[f['filename']])
                if d is not None:
                    dups.append(f"{f['filename']}: identical to {d.benchmark.slug}/{d.filename} (#{d.id})")
        if dups:
            update(lambda sub: (_set_step(sub, 'dedup', 'fail', '\n'.join(dups)), setattr(sub, 'status', 'failed')))
            _skip_rest(sub_id, after='dedup')
            return
        update(lambda sub: (_set_step(sub, 'dedup', 'pass', 'no identical implementation in the library'),
                            _set_step(sub, 'approval', 'pending', 'waiting for a maintainer'),
                            setattr(sub, 'status', 'awaiting_review')))
    except Exception:
        tb = traceback.format_exc()
        update(lambda sub: (setattr(sub, 'status', 'failed'),
                            _set_step(sub, 'simulation', 'fail', tb[-4000:])))


def _skip_rest(sub_id: int, after: str):
    names = [n for n, _ in PIPELINE_STEPS]
    later = names[names.index(after) + 1:]
    with session_scope() as s:
        sub = s.get(Submission, sub_id)
        for n in later:
            if (sub.steps.get(n) or {}).get('status') in (None, 'pending'):
                _set_step(sub, n, 'skipped', 'earlier step failed')


# ----------------------------------------------------------------------------
# Maintainer decision
# ----------------------------------------------------------------------------

def approve(sub_id: int, reason: str, actor: str = 'maintainer') -> str:
    """Publish the submission's files as implementations; returns the benchmark slug."""
    with session_scope() as s:
        sub = s.get(Submission, sub_id)
        if sub.status != 'awaiting_review':
            raise ValueError(f"submission is {sub.status}, not awaiting review")
        m = sub.manifest
        spec = service.describe_spec(m['module_type'], m['k_value'], m['bitwidth'], m['operations'], m.get('params'))
        if m.get('description'):
            spec['description'] = m['description']
        bm = service.get_or_create_benchmark(s, spec)
        gen = m.get('generator') or {}
        submitter = m.get('submitter') or {}
        added = []
        for f in m['files']:
            metrics = dict(sub.results.get(f['filename'], {}))
            metrics.pop('sha256', None)
            if metrics.get('verified_at'):
                metrics['verified_at'] = datetime.fromisoformat(metrics['verified_at'])
            impl = service.add_implementation(
                s, bm, f['language'], sub.files[f['filename']],
                source=m.get('source', 'human-authored'), provider=gen.get('provider') or None,
                model_requested=gen.get('model') or None, model_responded=gen.get('model') or None,
                prompt_hash=gen.get('prompt_sha256') or None, metrics=metrics, submitter=submitter,
                notes=f.get('notes', ''))
            if impl is not None:
                added.append(impl.id)
        service.add_review_event(s, bm, 'submitted', f"submission #{sub.id} by {sub.submitter_name}",
                                 actor=sub.submitter_name or 'submitter', submission_id=sub.id)
        service.add_review_event(s, bm, 'approved', reason, actor=actor, submission_id=sub.id)
        _set_step(sub, 'approval', 'pass', reason)
        sub.status = 'approved'
        sub.decision_reason = reason
        sub.decided_at = datetime.utcnow()
        return bm.slug


def reject(sub_id: int, reason: str, actor: str = 'maintainer'):
    with session_scope() as s:
        sub = s.get(Submission, sub_id)
        if sub.status not in ('awaiting_review', 'failed'):
            raise ValueError(f"submission is {sub.status}")
        _set_step(sub, 'approval', 'fail', reason)
        sub.status = 'rejected'
        sub.decision_reason = reason
        sub.decided_at = datetime.utcnow()


def list_submissions(session, status: Optional[str] = None) -> List[Submission]:
    stmt = select(Submission).order_by(Submission.created_at.desc())
    if status:
        stmt = stmt.where(Submission.status == status)
    return list(session.execute(stmt).scalars())


def get_by_token(session, token: str) -> Optional[Submission]:
    return session.execute(select(Submission).where(Submission.token == token)).scalar_one_or_none()
