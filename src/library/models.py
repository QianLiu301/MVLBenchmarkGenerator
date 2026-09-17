"""Library tables.

Benchmark      = the *specification* (what RevLib calls a "function"): module type,
                 radix k, bitwidth, algebraic structure, operations. One row per
                 unique spec, addressed by a stable slug such as `alu_k3_8t`.
Implementation = one code file realising a spec (RevLib's "circuit realization"),
                 with the metrics shown in the detail table: language, source,
                 generating model, LOC, test vectors, simulation and golden-model
                 verification results.
"""
from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Integer, JSON,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import relationship

from .db import Base

MODULE_TYPES = {
    'alu': 'ALU',
    'register': 'Register File',
    'cpu-risc-v': 'RISC-V CPU',
}
LANGUAGES = {'c': 'C', 'python': 'Python', 'verilog': 'Verilog', 'vhdl': 'VHDL'}
LANGUAGE_EXT = {'c': '.c', 'python': '.py', 'verilog': '.v', 'vhdl': '.vhd'}
SOURCES = {
    'llm-generated': 'LLM-generated',
    'human-authored': 'Human-authored',
    'reference': 'Reference',
}
LICENSE_ID = 'CC-BY-4.0'


class Benchmark(Base):
    __tablename__ = 'benchmarks'

    id = Column(Integer, primary_key=True)
    slug = Column(String(64), unique=True, nullable=False, index=True)   # alu_k3_8t
    module_type = Column(String(32), nullable=False, index=True)          # alu / register / cpu-risc-v
    k_value = Column(Integer, nullable=False, index=True)
    bitwidth = Column(Integer, nullable=False, index=True)
    logic_type = Column(String(64), nullable=False)                       # "GF(3)", "GF(2^2)", "Z/6Z"
    logic_family = Column(String(16), nullable=False, index=True)         # gf-prime / gf-ext / ring
    mod_value = Column(Integer, nullable=False)
    operations = Column(JSON, nullable=False, default=list)               # ["ADD", "SUB", ...]
    params = Column(JSON, nullable=False, default=dict)                   # register_count, pipeline_stages
    title = Column(String(160), nullable=False)
    description = Column(Text, nullable=False, default='')
    status = Column(String(16), nullable=False, default='published', index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    download_count = Column(Integer, nullable=False, default=0)

    implementations = relationship('Implementation', back_populates='benchmark',
                                   cascade='all, delete-orphan',
                                   order_by='Implementation.language')

    __table_args__ = (UniqueConstraint('slug', name='uq_benchmark_slug'),)

    @property
    def module_label(self):
        return MODULE_TYPES.get(self.module_type, self.module_type)

    @property
    def published_implementations(self):
        return [i for i in self.implementations if i.status == 'published']


class Implementation(Base):
    __tablename__ = 'implementations'

    id = Column(Integer, primary_key=True)
    benchmark_id = Column(Integer, ForeignKey('benchmarks.id', ondelete='CASCADE'),
                          nullable=False, index=True)
    language = Column(String(16), nullable=False, index=True)
    filename = Column(String(128), nullable=False)
    code = Column(Text, nullable=False)
    source = Column(String(24), nullable=False, default='llm-generated', index=True)
    provider = Column(String(32))            # deepseek / gemini / ...
    model_requested = Column(String(96))     # what we asked for
    model_responded = Column(String(96))     # what the API reported back
    generated_at = Column(DateTime)

    # Metrics (RevLib: lines / gates / costs)
    loc = Column(Integer, nullable=False, default=0)
    test_vectors = Column(Integer, nullable=False, default=0)
    sim_status = Column(String(16), nullable=False, default='unverified')      # pass / fail / unverified
    sim_passed = Column(Integer, nullable=False, default=0)
    sim_total = Column(Integer, nullable=False, default=0)
    golden_status = Column(String(16), nullable=False, default='unverified')   # PASS / LOGIC_ERROR / ... / unverified
    golden_passed = Column(Integer, nullable=False, default=0)
    golden_compared = Column(Integer, nullable=False, default=0)
    verification_report = Column(JSON)       # full ValidationReport.summary()

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
