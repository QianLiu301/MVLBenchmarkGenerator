"""Copy the library from the bundled SQLite file into the Postgres database in DATABASE_URL.

Used on 2026-10-01 to move the site back to Neon: the SQLite file had become the
complete copy (56 specifications, 654 implementations) while Neon still held the
state of 2026-09-22.

    python scripts/sqlite_to_postgres.py            # compare only, change nothing
    python scripts/sqlite_to_postgres.py --write    # back up Postgres, then copy

The SQLite file was removed from the repository afterwards; to run this again,
restore it from history first: git checkout be8dba5 -- data/library.db

--write first saves every Postgres table to output/backups/ as JSON. It then
replaces benchmarks, implementations, review_events and news in one transaction,
so a failure leaves Postgres as it was. Submissions are kept: SQLite has none,
and the ones in Postgres are not reproduced anywhere else.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import dotenv_values                                  # noqa: E402
from sqlalchemy import create_engine, func, select, text          # noqa: E402
from sqlalchemy.pool import NullPool                              # noqa: E402

from library.db import Base                                       # noqa: E402
from library import models                                        # noqa: E402,F401  registers the tables

COPIED = ['benchmarks', 'implementations', 'review_events', 'news']   # parents before children
KEPT = ['submissions']


def postgres_url() -> str:
    import os
    url = (os.environ.get('DATABASE_URL') or dotenv_values(PROJECT_ROOT / '.env').get('DATABASE_URL') or '').strip()
    if not url.startswith(('postgres://', 'postgresql://')):
        sys.exit('DATABASE_URL is not a Postgres URL')
    return 'postgresql://' + url.split('://', 1)[1]


def counts(engine):
    with engine.connect() as c:
        return {t: c.execute(select(func.count()).select_from(Base.metadata.tables[t])).scalar()
                for t in COPIED + KEPT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='back up Postgres, then copy SQLite into it')
    args = ap.parse_args()

    lite = create_engine(f"sqlite:///{(PROJECT_ROOT / 'data' / 'library.db').as_posix()}")
    # no pool: a serverless Postgres only suspends while nothing is connected
    pg = create_engine(postgres_url(), poolclass=NullPool, connect_args={'connect_timeout': 15})
    Base.metadata.create_all(pg)

    before = counts(pg)
    print(f"{'table':16s} {'sqlite':>7s} {'postgres':>9s}")
    for t, n in counts(lite).items():
        print(f'{t:16s} {n:7d} {before[t]:9d}')
    if not args.write:
        print('\ncompare only; run with --write to copy')
        return

    # 1. back up everything Postgres holds now
    backup_dir = PROJECT_ROOT / 'output' / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"postgres_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    with pg.connect() as c:
        dump = {t: [dict(r._mapping) for r in c.execute(select(Base.metadata.tables[t]))]
                for t in COPIED + KEPT}
    backup.write_text(json.dumps(dump, default=str, ensure_ascii=False), encoding='utf-8')
    print(f'\nbackup: {backup} ({sum(len(v) for v in dump.values())} rows)')

    # 2. read SQLite through the model tables, so JSON and datetime columns arrive typed
    with lite.connect() as c:
        rows = {t: [dict(r._mapping) for r in c.execute(select(Base.metadata.tables[t]))] for t in COPIED}

    # 3. replace the copied tables in one transaction
    with pg.begin() as c:
        c.execute(text('TRUNCATE ' + ', '.join(COPIED) + ' RESTART IDENTITY'))
        for t in COPIED:
            if rows[t]:
                c.execute(Base.metadata.tables[t].insert(), rows[t])
            # ids were copied explicitly, so move each sequence past the largest one
            c.execute(text(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), "
                           f"COALESCE((SELECT MAX(id) FROM {t}), 0) + 1, false)"))

    after = counts(pg)
    expected = {**{t: len(rows[t]) for t in COPIED}, **{t: before[t] for t in KEPT}}
    print(f"\n{'table':16s} {'postgres':>9s} {'expected':>9s}")
    for t in COPIED + KEPT:
        print(f'{t:16s} {after[t]:9d} {expected[t]:9d}{"" if after[t] == expected[t] else "  <-- MISMATCH"}')
    if after != expected:
        sys.exit('counts differ')
    print('copied')


if __name__ == '__main__':
    main()
