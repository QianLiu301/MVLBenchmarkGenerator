"""Library tables.

Benchmark      = the *specification* (what RevLib calls a "function"): module type,
                 radix k, bitwidth, algebraic structure, operations. One row per
                 unique spec, addressed by a stable slug such as `alu_k3_8t`.
Implementation = one code file realising a spec (RevLib's "circuit realization"),
                 with metrics, a verification record and provenance.
Submission     = an uploaded manifest + files moving through the review pipeline
                 (schema → lint → simulation → formal → llm review → dedup →
                 maintainer approval). Approval turns it into Implementation rows.
ReviewEvent    = audit trail shown as "Review history" on the detail page.
"""
from datetime import datetime

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, JSON, String,
                        Text, UniqueConstraint)
from sqlalchemy.orm import relationship

from .db import Base

MODULE_TYPES = {
    'alu': 'ALU',
    'register': 'Register File',
    'cpu-risc-v': 'RISC-V CPU',
}
MODULE_ICONS = {'alu': 'cpu', 'register': 'database', 'cpu-risc-v': 'microchip'}
LANGUAGES = {'c': 'C', 'python': 'Python', 'verilog': 'Verilog', 'vhdl': 'VHDL'}
LANGUAGE_EXT = {'c': '.c', 'python': '.py', 'verilog': '.v', 'vhdl': '.vhd'}
SOURCES = {
    'llm-generated': 'LLM-generated',
    'human-authored': 'Human-authored',
    'reference': 'Reference',
}
LICENSE_ID = 'CC-BY-4.0'
GOLDEN_MODEL_VERSION = '1.0'

# Pipeline steps, in order. Each Submission stores {step: {status, log, detail}}.
PIPELINE_STEPS = [
    ('schema', 'Schema'),
    ('lint', 'Lint'),
    ('simulation', 'Simulation vs golden'),
    ('formal', 'Formal (optional)'),
    ('llm_review', 'LLM review'),
    ('dedup', 'Dedup'),
    ('approval', 'Maintainer approval'),
]
STEP_STATES = ('pending', 'running', 'pass', 'fail', 'skipped')


class Benchmark(Base):
    __tablename__ = 'benchmarks'

    id = Column(Integer, primary_key=True)
    slug = Column(String(64), unique=True, nullable=False, index=True)   # alu_k3_8t
    module_type = Column(String(32), nullable=False, index=True)
    k_value = Column(Integer, nullable=False, index=True)
    bitwidth = Column(Integer, nullable=False, index=True)
    logic_type = Column(String(64), nullable=False)                       # "GF(3)", "GF(2^2)", "mod 6"
    logic_family = Column(String(16), nullable=False, index=True)         # gf-prime / gf-ext / ring
    mod_value = Column(Integer, nullable=False)
    operations = Column(JSON, nullable=False, default=list)
    params = Column(JSON, nullable=False, default=dict)
    title = Column(String(160), nullable=False)
    description = Column(Text, nullable=False, default='')
    status = Column(String(16), nullable=False, default='published', index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    download_count = Column(Integer, nullable=False, default=0)

    implementations = relationship('Implementation', back_populates='benchmark',
                                   cascade='all, delete-orphan',
                                   order_by='Implementation.language')
    review_events = relationship('ReviewEvent', back_populates='benchmark',
                                 cascade='all, delete-orphan', order_by='ReviewEvent.created_at')

    __table_args__ = (UniqueConstraint('slug', name='uq_benchmark_slug'),)

    @property
    def module_label(self):
        return MODULE_TYPES.get(self.module_type, self.module_type)

    @property
    def published_implementations(self):
        return [i for i in self.implementations if i.status == 'published']

    def language_status(self):
        """{language: 'verified' | 'pending' | 'unverified'} for the browse badges."""
        out = {}
        for i in self.published_implementations:
            cur = out.get(i.language)
            st = 'verified' if i.golden_status == 'PASS' else ('pending' if i.golden_status == 'unverified' else 'unverified')
            rank = {'verified': 2, 'pending': 1, 'unverified': 0}
            if cur is None or rank[st] > rank[cur]:
                out[i.language] = st
        return out


class Implementation(Base):
    __tablename__ = 'implementations'

    id = Column(Integer, primary_key=True)
    benchmark_id = Column(Integer, ForeignKey('benchmarks.id', ondelete='CASCADE'),
                          nullable=False, index=True)
    language = Column(String(16), nullable=False, index=True)
    filename = Column(String(128), nullable=False)
    code = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)
    source = Column(String(24), nullable=False, default='llm-generated', index=True)
    provider = Column(String(32))
    model_requested = Column(String(96))
    model_responded = Column(String(96))
    prompt_hash = Column(String(64))          # sha256 of the generation prompt (LLM sources)
    generated_at = Column(DateTime)

    # Metrics
    loc = Column(Integer, nullable=False, default=0)
    test_vectors = Column(Integer, nullable=False, default=0)
    sim_status = Column(String(16), nullable=False, default='unverified')      # pass / fail / unverified
    sim_passed = Column(Integer, nullable=False, default=0)
    sim_total = Column(Integer, nullable=False, default=0)
    golden_status = Column(String(16), nullable=False, default='unverified')   # PASS / LOGIC_ERROR / ... / unverified
    golden_passed = Column(Integer, nullable=False, default=0)
    golden_compared = Column(Integer, nullable=False, default=0)

    # Verification record
    verification_strength = Column(String(48))   # "random(N=50, seed=42)" / "exhaustive" / "bounded-formal"
    verification_meta = Column(JSON)             # {tools: {gcc: "15.2.0"...}, golden_model: "1.0", strategies: {...}}
    verification_report = Column(JSON)           # ValidationReport.summary() (strategy A) + strategy B summary
    verification_log = Column(Text)              # raw simulator output (truncated)
    verified_at = Column(DateTime)

    # Provenance
    submitter_name = Column(String(120))
    submitter_email = Column(String(160))
    submitter_affiliation = Column(String(200))
    notes = Column(Text, default='')
    license = Column(String(24), nullable=False, default=LICENSE_ID)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default='published', index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    download_count = Column(Integer, nullable=False, default=0)

    benchmark = relationship('Benchmark', back_populates='implementations')

    @property
    def language_label(self):
        return LANGUAGES.get(self.language, self.language)

    @property
    def source_label(self):
        return SOURCES.get(self.source, self.source)

    @property
    def verified(self) -> bool:
        return self.golden_status == 'PASS'

    @property
    def state(self) -> str:
        """verified / pending / unverified — the three legend states."""
        if self.golden_status == 'PASS':
            return 'verified'
        if self.golden_status == 'unverified':
            return 'pending'
        return 'unverified'


class Submission(Base):
    __tablename__ = 'submissions'

    id = Column(Integer, primary_key=True)
    token = Column(String(32), unique=True, nullable=False, index=True)   # public status URL
    manifest = Column(JSON, nullable=False)
    files = Column(JSON, nullable=False)          # {filename: code}
    slug = Column(String(64), index=True)         # derived from manifest
    submitter_name = Column(String(120))
    submitter_email = Column(String(160))
    submitter_affiliation = Column(String(200))
    status = Column(String(16), nullable=False, default='queued', index=True)  # queued / running / awaiting_review / approved / rejected / failed
    steps = Column(JSON, nullable=False, default=dict)  # {step: {status, log, detail, finished_at}}
    results = Column(JSON)                        # per-file verification metrics
    decision_reason = Column(Text)
    decided_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def files_summary(self):
        return ', '.join(sorted(self.files.keys()))


class ReviewEvent(Base):
    __tablename__ = 'review_events'

    id = Column(Integer, primary_key=True)
    benchmark_id = Column(Integer, ForeignKey('benchmarks.id', ondelete='CASCADE'), index=True)
    submission_id = Column(Integer, ForeignKey('submissions.id', ondelete='SET NULL'), nullable=True)
    implementation_id = Column(Integer, nullable=True)
    action = Column(String(24), nullable=False)   # seeded / submitted / auto-checked / approved / rejected / reverified
    actor = Column(String(64), nullable=False, default='system')
    detail = Column(Text, default='')
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    benchmark = relationship('Benchmark', back_populates='review_events')
