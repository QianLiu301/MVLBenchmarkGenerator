# Story line — The MVL Benchmark Library: An Online Resource for Multi-Valued Benchmarks

## Introduction

multi-valued logic is pursued because it promises higher information density, fewer interconnects and lower power consumption than binary logic \cite{S:81}

whether these promises hold is decided by comparison, and a comparison is only as good as the designs it is carried out on

for binary logic common benchmark sets have been available since the early 80s \cite{BF:85,BD:99}, and other domains built such a resource as well, for reversible circuits \cite{WGT+:2008}, satisfiability \cite{HS:2000} and the traveling salesman problem \cite{R:1991}

for multi-valued logic no such resource exists; that no standard benchmark functions are available for comparing MVL designs was stated twenty years ago \cite{TB:05}, and it is still the case

instead each community evaluates on its own material: randomly generated two-variable functions in synthesis \cite{XX:2024}, encoded binary circuits for decision diagrams \cite{BK:1999}, and a binary ISA as the reference for a 32-trit ternary architecture \cite{BBMG:2025}

this material is rarely distributed, so published comparisons rest on designs that other researchers cannot obtain

producing meaningful MVL designs by hand is expensive, but recently it has been shown that they can be generated with very low effort using LLMs \cite{D:2026}

designs alone, however, are not benchmarks: what a design computes has to be stated, and that it computes it has to be established

in MVL the name of a design does not determine its function: an "8-trit ALU over GF(3)" may denote the ring Z/3^8 Z, the field GF(3^8) or digit-wise arithmetic in GF(3)[x]/(x^8), ternary designs are built on unbalanced as well as on balanced digits \cite{BBMG:2025}, and there is no convention for the status flags as there is for two's complement

and a generated design comes with its own testbench which it passes, so its test output is no evidence of correctness

in this paper we present the MVL Benchmark Library, an online resource in which every entry states the algebra it computes in, and carries the record of its verification against an independent reference model

## Contributions

a format in which a multi-valued benchmark is unambiguous: the algebra, the formula of every operation and the rule for every status flag are part of the entry, so that two entries carrying the same name denote the same function

a reference model independent of whoever produced an implementation, and validated itself against arbitrary-precision integer arithmetic, a third-party Galois field library, the ring and field axioms, and the binary special case --- a validation that found a real defect in our own polynomial reduction

a verification record published with every entry: the implementation is compared to the reference model both on the vectors it prints itself and on injected vectors after its test section has been replaced, exhaustively where the operand space allows, and the record names the tool versions, vector counts, seed and checksum, so that anyone can repeat it

the resource itself: 56 specifications over radices 2 to 9 and 8 to 14 digits, 654 implementations in C, Python, Verilog and VHDL of which 65% are verified, implementations that do not pass kept with their reports and the model that produced them, all entries of a specification sharing one interface and one reference model, archived under a DOI and open to submissions checked by the same procedure

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
