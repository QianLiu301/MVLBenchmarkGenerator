"""Database engine and session factory.

DATABASE_URL set   -> Postgres (a hosted database). Connections are NOT pooled: a
                      serverless Postgres only suspends its compute while nothing is
                      connected, and a pool held open by a long-running web process
                      keeps it awake around the clock. That is what exhausted the Neon
                      free allowance on 2026-09-22 — the traffic was negligible, the
                      idle connection was not.
DATABASE_URL unset -> SQLite at data/library.db, which ships with the application.
                      The library is read-mostly (browsing, downloads, API), so the
                      file serves it well and has no quota. Writes (submissions,
                      approvals, download counters) last only until the next deploy,
                      because the container filesystem is replaced.

History: Neon ran out of compute hours on 2026-09-22 (an idle pooled connection,
fixed with NullPool) and of network transfer on 2026-10-02 (list pages loaded the
code and logs of every implementation, fixed by deferring those columns in
models.py). Each time the site went back to the bundled file.
"""
import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

_engine = None
_Session = None


def _database_url() -> str:
    url = os.environ.get('DATABASE_URL', '').strip()
    if url:
        # Whatever form the URL is pasted in -- "postgres://" (Heroku style, refused by
        # SQLAlchemy 2.x), "postgresql://", or "postgresql+psycopg://" as Neon's console
        # offers it, which needs psycopg 3 -- use the driver that is installed, psycopg2.
        scheme, sep, rest = url.partition('://')
        if sep and scheme.split('+')[0] in ('postgres', 'postgresql'):
            url = 'postgresql+psycopg2://' + rest
        return url
    return f"sqlite:///{_sqlite_path().as_posix()}"


def _sqlite_path() -> Path:
    data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    return data_dir / 'library.db'


def get_engine():
    global _engine, _Session
    if _engine is None:
        url = _database_url()
        missing = url.startswith('sqlite') and not _sqlite_path().exists()
        kwargs = {'pool_pre_ping': True, 'future': True}
        if url.startswith('sqlite'):
            kwargs['connect_args'] = {'check_same_thread': False}
        else:
            # no idle connections, so a serverless Postgres can suspend between requests
            kwargs['poolclass'] = NullPool
            kwargs['connect_args'] = {'connect_timeout': 10}
        _engine = create_engine(url, **kwargs)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        backend = 'postgres' if url.startswith('postgresql') else 'sqlite'
        print(f"[library] database: {backend}")
        if missing:
            # no Postgres and no bundled file: the site is about to serve an empty library
            print('[library] *** WARNING: neither DATABASE_URL nor data/library.db; '
                  'the library is EMPTY', flush=True)
    return _engine


def init_db():
    """Create tables if they don't exist (idempotent)."""
    from . import models  # noqa: F401 — registers tables on Base
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope():
    get_engine()
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
