"""MVL Benchmark Library — persistent store of benchmark specs and their implementations.

Layout:
  db.py       engine / session (Postgres via DATABASE_URL, SQLite fallback for local dev)
  models.py   Benchmark (spec) and Implementation (one code file bound to a spec)
  service.py  queries, slugs, zip/BibTeX builders, ingest of generated + validated code
"""
