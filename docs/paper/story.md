# Story line — The MVL Benchmark Library: An Online Resource for Multi-Valued Benchmarks

## Introduction

benchmarking is highly relevant for the evaluation and comparison of newly proposed EDA algorithms

what makes a benchmark useful is not that it exists, but that different groups use the same one; only then are published results comparable

for binary logic such common sets have been available since the early 80s \cite{BF:85,BD:99}; proposed for testing algorithms, they were later used intensively for synthesis and verification as well

also in other domains a common resource became the reference for a whole community, like reversible circuits \cite{WGT+:2008}, satisfiability \cite{HS:2000} and the traveling salesman problem \cite{R:1991}

for multi-valued logic no such resource exists, although the field is active; ISMVL 2025 alone had more than 40 papers on synthesis, decision diagrams, architectures, emerging devices and security

instead each community evaluates on its own material: randomly generated two-variable functions in synthesis \cite{XX:2024}, encoded binary circuits for decision diagrams \cite{BK:1999}, self-designed cells at device level, and a binary ISA as the reference for a 32-trit ternary architecture \cite{BBMG:2025}

this material is mostly not distributed --- the random sets are regenerated per paper without a seed --- so published comparisons rest on sets that are not the same set

what these substitutes have in common is that they are cheap to obtain; a common resource instead requires a body of designs, and producing meaningful MVL designs by hand is expensive

recently it has been shown that meaningful MVL designs can be generated with very low effort using LLMs \cite{D:2026}, which removes exactly this obstacle

however, cheap production alone does not create a shared resource: what a design computes has to be stated, and that it computes it has to be established

in MVL the name of a design does not determine its function: an "8-trit ALU over GF(3)" may denote the ring Z/3^8 Z, the field GF(3^8), or digit-wise arithmetic in GF(3)[x]/(x^8), and ternary designs are built on unbalanced as well as on balanced digits \cite{BBMG:2025}

also there is no convention for the status flags, as there is for negative numbers in two's complement

and while designs were expensive, their cost was itself a filter on quality; a generated design passes no such filter, and it comes with its own testbench which it passes, so its test output is no evidence of correctness

in this paper we present the MVL Benchmark Library, an online resource in which every entry states the algebra it computes in, and carries the record of its verification against an independent reference model

currently 56 specifications with 654 implementations in C, Python, Verilog and VHDL are provided, of which 65% are verified; implementations that do not pass are kept with their reports, and the model that produced an implementation is recorded

all entries of a specification share one interface and are judged by one reference model, so a single testbench drives any of them, the radix can be varied with everything else held fixed, and four description languages of one specification can be compared

a file format, a reference model, a submission procedure and a JSON API are part of the resource, and the library is versioned and archived under a DOI, so it can be cited and can grow

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
