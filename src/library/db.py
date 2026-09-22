"""Database engine and session factory.

DATABASE_URL set   -> Postgres (a hosted database). Connections are NOT pooled: a
                      serverless Postgres only suspends its compute while nothing is
                      connected, and a pool held open by a long-running web process
                      keeps it awake around the clock. That is what exhausted the Neon
                      free allowance on 2026-09-22 — the traffic was negligible, the
                      idle connection was not.
DATABASE_URL unset -> SQLite at data/library.db. The file ships with the application,
                      which suits a read-mostly library: browsing, downloads and the
                      API need no writes. Writes (submissions, approvals, download
                      counters) live only until the next deploy, because the container
                      filesystem is replaced.
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
        # Heroku/Neon style "postgres://" is not accepted by SQLAlchemy 2.x
        if url.startswith('postgres://'):
            url = 'postgresql://' + url[len('postgres://'):]
        return url
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    return f"sqlite:///{(data_dir / 'library.db').as_posix()}"


def get_engine():
    global _engine, _Session
    if _engine is None:
        url = _database_url()
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
