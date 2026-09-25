# Story line — The MVL Benchmark Library: An Online Resource for Multi-Valued Benchmarks

## Introduction

benchmarking is highly relevant for the evaluation and comparison of newly proposed EDA algorithms

evaluation on random functions is not sufficient, but structures occurring in designs, like adders, multipliers or {\em Arithmetic Logical Units} (ALUs) should be considered

for binary logic such benchmark sets have been available since the early 80s \cite{BF:85,BD:99}

originally proposed for testing algorithms, they were later used intensively for synthesis and verification as well, so a benchmark set is used for purposes it was not built for

also in other domains benchmarks are established, like reversible circuits \cite{WGT+:2008}

for multi-valued logic circuits very few benchmark sets are available; often binary circuits have been used with encodings, but these do not reflect the MVL nature \cite{BK:1999}

in \cite{RS:2018} an approach based on DDs was proposed; this allows for circuit generation, but on very low levels only

recently it has been shown that based on LLMs meaningful MVL benchmarks can be generated with very low effort \cite{D:2026}, following the use of LLMs in other fields of EDA \cite{JHQ+:2025,FFKR:2024}

however, a generated design is not yet a benchmark: to be used by others it has to be unambiguous, and its correctness has to be established

in MVL the name of a design does not determine its function: an "8-trit ALU over GF(3)" may denote the ring Z/3^8 Z, the field GF(3^8), or digit-wise arithmetic in GF(3)[x]/(x^8)

also there is no convention for the status flags, as there is for negative numbers in two's complement

moreover, a generated design comes with its own testbench which it passes, so its test output is no evidence of correctness

in this paper we present the MVL Benchmark Library, an online resource in which every entry states the algebra it computes in, and carries the record of its verification against an independent reference model

currently 56 specifications with 654 implementations in C, Python, Verilog and VHDL are provided, of which 65% are verified; the model that produced an implementation is recorded, and implementations that do not pass are kept as well

## The paper then contains

The MVL Benchmark Library --- the parameterised family, what it contains, and how it is obtained and cited

Benchmark format --- the two algebraic families, the formula of every operation and the rule for every status flag

Verification --- the independent reference model and its own validation, the two comparisons applied to every implementation, and the record published with each entry

Using the library --- the common interface of the HDL entries, browsing, archives, API, DOI, and the submission process

Conclusion and future work --- further module types, true extension fields, further description languages and lower abstraction levels, following \cite{D:2026}

---

## Notes — not part of what is sent

One observation the collection already permits, for Section V: the same specification is
implemented correctly in C and Python in almost every case (96% and 99%), while the hardware
description languages are far less reliable (Verilog 55%, VHDL 28%); for one of the two
producers covering the complete grid, only one of 54 VHDL implementations passed.

Not claimable: an effect of algebraic family or radix. The two complete grids point in
opposite directions (modular vs field: 71% / 75% for one producer, 61% / 51% for the other),
and radix shows no consistent trend.
