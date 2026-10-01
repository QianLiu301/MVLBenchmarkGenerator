"""Database engine and session factory.

DATABASE_URL set   -> Postgres (a hosted database). Connections are NOT pooled: a
                      serverless Postgres only suspends its compute while nothing is
                      connected, and a pool held open by a long-running web process
                      keeps it awake around the clock. That is what exhausted the Neon
                      free allowance on 2026-09-22 — the traffic was negligible, the
                      idle connection was not.
DATABASE_URL unset -> an empty SQLite file at data/library.db, for local development
                      and tests. The library itself lives in Postgres; from 2026-09-22
                      to 2026-10-01 a copy shipped with the app while the Neon quota
                      was exhausted, and was removed once the site ran on Neon again.
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
        if backend == 'sqlite' and os.environ.get('RENDER'):
            # Render sets RENDER; there, SQLite means DATABASE_URL is missing and the
            # site is about to serve an empty library
            print('[library] *** WARNING: DATABASE_URL is not set on Render; '
                  'the site is running on an EMPTY database', flush=True)
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
