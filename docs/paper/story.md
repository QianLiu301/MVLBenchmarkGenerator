# Story line — The MVL Benchmark Library: An Online Resource for Multi-Valued Benchmarks

## Introduction

multi-valued logic is pursued because it promises higher information density, fewer interconnects and lower power consumption than binary logic \cite{S:81}

whether these promises hold is decided by comparison, and a comparison is only as good as the designs it is carried out on

for binary logic common benchmark sets have been available since the early 80s \cite{BF:85,BD:99}, and other domains built such a resource as well, for reversible circuits \cite{WGT+:2008}, satisfiability \cite{HS:2000} and the traveling salesman problem \cite{R:1991}

for multi-valued logic no such resource exists; that no standard benchmark functions are available for comparing MVL designs was stated twenty years ago \cite{TB:05}, and it is still the case

instead each community evaluates on its own material: randomly generated two-variable functions in synthesis \cite{XX:2024}, binary circuits with grouped and encoded variables, which do not reflect the MVL nature \cite{BK:1999}, and a binary ISA as the reference for a 32-trit ternary architecture \cite{BBMG:2025}

this material is rarely distributed, so published comparisons rest on designs that other researchers cannot obtain

producing meaningful MVL designs by hand is expensive, but recently it has been shown that they can be generated with very low effort using LLMs \cite{D:2026}

designs alone, however, are not benchmarks: what a design computes has to be stated, and that it computes it has to be established

in MVL the name of a design does not determine its function: a ternary ALU may be built on unbalanced digits 0, 1, 2 or on balanced digits -1, 0, 1 \cite{BBMG:2025}, and there is no convention for its status flags as there is for two's complement

and a generated design comes with its own testbench which it passes, so its test output is no evidence of correctness

in this paper we present the MVL Benchmark Library, an online resource in which every entry states the algebra it computes in, and carries the record of its verification against an independent reference model

## Contributions

a format in which a multi-valued benchmark is unambiguous: the algebra, the formula of every operation and the rule for every status flag are part of the entry, so that two entries carrying the same name denote the same function

a reference model independent of whoever produced an implementation, and validated itself against arbitrary-precision integer arithmetic, a third-party Galois field library, the ring and field axioms, and the binary special case --- a validation that found a real defect in our own polynomial reduction

a verification record published with every entry: the implementation is compared to the reference model both on the vectors it prints itself and on injected vectors after its test section has been replaced, exhaustively where the operand space allows, and the record names the tool versions, vector counts, seed and checksum, so that anyone can repeat it

the resource itself: 56 specifications over radices 2 to 9 and 8 to 14 digits, 654 implementations in C, Python, Verilog and VHDL of which 65% are verified, implementations that do not pass kept with their reports and the model that produced them, all entries of a specification sharing one interface and one reference model, archived under a DOI and open to submissions checked by the same procedure

## The paper then contains

The MVL Benchmark Library
  a parameterised family rather than a fixed list: module type x radix k x digit count n x algebraic family
  currently one module type, the ALU, with the six operations ADD, SUB, MUL, NEG, INC and DEC, for k = 2 to 9 and n = 8 to 14 digits
  names are derived from the parameters, so adding a radix or a width adds entries without changing anything else
  how an entry is obtained and cited: browsing and filtering, per-entry and whole-library archives, a JSON API, a DOI-archived release, CC BY 4.0

Benchmark format
  family M, the ring Z/k^n Z: radix-k integers, arithmetic modulo k^n, carry and borrow defined, and the negative flag defined explicitly as result >= k^n / 2, since MVL has no two's complement to inherit a convention from
  family F, the ring GF(q)[x]/(x^n): every digit an element of GF(q), the operations digit-wise, multiplication the polynomial product truncated to n digits under an irreducible polynomial fixed per q, hence no carry between digits and no negative flag
  the formula of every operation and the rule for every status flag are part of the entry, because the two families share the same informal names

Verification
  the reference model is derived from the definitions of the format; it uses no language model, no entry of the library and no simulator, it is versioned, and every record names the version it was produced with
  it is validated against four things from outside the project: arbitrary-precision integer arithmetic; an independent Galois field package, table by table, under the same irreducible polynomial; the ring and field axioms directly, including SUB after ADD = identity and NEG = SUB(0, .); and the requirement that k = 2 reproduce an ordinary binary ALU
  this validation found a defect of our own: the polynomial reduction used the irreducible polynomial in reverse order, which affects GF(8) and GF(16), and the library contains k = 8
  first comparison: every vector the implementation prints itself is recomputed by the reference model and compared, result and all flags
  second comparison: the implementation's test section is replaced by a generated driver fed with reference vectors, exhaustively over all operand pairs where k^n <= 256, otherwise a seeded sample together with every edge case
  both have to agree completely; the first alone proves nothing, as 24 implementations once passed it while their injected driver had silently failed to build
  the record published with an entry names the tool versions, the vectors compared and passed per comparison, the strength of the check, the file checksum and the reference model version, so that anyone can repeat it

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
