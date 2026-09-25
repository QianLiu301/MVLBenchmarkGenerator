# Story line — The MVL Benchmark Library: An Online Resource for Multi-Valued Benchmarks

## Introduction

- benchmarking is essential for the evaluation and comparison of newly proposed EDA algorithms;
  for binary logic common sets have been available since the early 80s \cite{BF:85,BD:99}, and
  dedicated benchmarks exist in adjacent domains such as reversible circuits \cite{WGT+:2008}
- for multi-valued logic very few benchmark sets are available; encoded binary circuits do not
  reflect the MVL nature \cite{BK:1999}, and DD-based generation yields circuits on very low
  levels only \cite{RS:2018}
- recently it was shown that meaningful MVL benchmarks can be generated on a high abstraction
  level using LLMs with very low effort \cite{D:2026}
- however, a generated design is not yet a benchmark: to be used by others it has to be
  unambiguous, and its correctness has to be established
- in MVL the name of a design does not determine its function: an "8-trit ALU over GF(3)" may
  denote the ring Z/3^8 Z, the field GF(3^8), or digit-wise arithmetic in GF(3)[x]/(x^8), and
  status flags such as *negative* have no convention as they have in two's complement
- furthermore, generated designs are delivered with a testbench which they pass, so self-reported
  test output is no evidence of correctness
- in this paper we present the MVL Benchmark Library, an online resource in which every entry
  states its algebraic structure and operation semantics, and carries the record of its
  verification against an independent reference model
- the library currently provides 56 specifications for radices 2 to 9 and 8 to 14 digits, each
  implemented in C, Python, Verilog and VHDL, resulting in 654 implementations of which 65% are
  verified
- every implementation records the model that produced it and is kept even when it does not pass,
  so the library documents not only the benchmarks but also which producers succeed on which
  specifications

## The paper then contains

- **The MVL Benchmark Library** — the parameterised family, what it contains, and what an entry
  consists of
- **Benchmark format** — the two algebraic families, the formula of every operation and the rule
  for every status flag
- **Verification** — the independent reference model and its own validation, the two comparisons
  applied to every implementation, and the record published with each entry
- **Using the library** — browsing, archives, API, DOI, the common interface of the HDL entries,
  and the submission process
- **Conclusion and future work** — further module types, true extension fields, further
  description languages and lower abstraction levels, following \cite{D:2026}

One observation the collection already permits: the same specification is implemented correctly
in C and Python in almost every case (96% and 99%), while the hardware description languages are
far less reliable (Verilog 55%, VHDL 28%) — for one of the two producers covering the complete
grid, only one of 54 VHDL implementations passed.
