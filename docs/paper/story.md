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
  an entry is not picked but addressed: its name is its parameters, so alu_k3_8t is the ALU with three logic values and eight digits
  the library currently holds one module type, the ALU, with the operations ADD, SUB, MUL, NEG, INC and DEC
  the radix runs from k = 2 to k = 9 and the operand width from n = 8 to n = 14 digits, which gives 56 specifications
  each specification is implemented in C, Python, Verilog and VHDL by several producers, which gives 654 implementations
  because only one parameter changes between neighbouring entries, the radix can be varied while the operations, the width and the reference model stay fixed; a list of hand-built circuits cannot offer this
  an entry is obtained by browsing and filtering, as a per-entry or whole-library archive, or through a JSON API, and the release is archived under a DOI and licensed CC BY 4.0

Benchmark format
  how the n digits of an operand relate to one another is not fixed by the radix alone, and no single rule covers k = 2 to 9
  a field GF(k) exists only when k is a prime power: for k = 6 there is none, so the digits can only be read as one integer in base 6
  for k = 4, 8 and 9 a field does exist, and its addition and multiplication are not arithmetic modulo k --- in GF(4), 1 + 1 = 0 rather than 2
  one family can therefore not cover the whole range, and the library defines two, with every entry stating which one it belongs to
  family M, the ring Z/k^n Z: the n digits form one number in base k, addition carries from digit to digit, the result is taken modulo k^n, carry and borrow are defined, and the negative flag is defined explicitly as result >= k^n / 2, since MVL inherits no convention from two's complement
  family F, the ring GF(q)[x]/(x^n): each of the n digits is an element of GF(q), the operations act on each digit separately so that no carry ever crosses a digit boundary, multiplication is the polynomial product truncated to n digits under an irreducible polynomial fixed per q, and there is no negative flag
  the difference is not notational: for a two-digit quaternary ALU, ADD(9, 7) is 0 with the carry set in family M and 14 with the carry clear in family F
  the formula of every operation and the rule for every status flag are therefore part of the entry, not of the prose around it

Verification
  a generated file contains both the design and the test that judges it, so its own test output cannot establish that it is correct
  the reference model is derived from the definitions of the format; it uses no language model, no entry of the library and no simulator, it is versioned, and every record names the version used
  it is validated against four things from outside the project: arbitrary-precision integer arithmetic; an independent Galois field package, table by table, under the same irreducible polynomial; the ring and field axioms directly, including SUB after ADD = identity and NEG = SUB(0, .); and the requirement that k = 2 reproduce an ordinary binary ALU
  this validation found a defect of our own: the polynomial reduction used the irreducible polynomial in reverse order, which affects GF(8) and GF(16), and the library contains k = 8
  the first comparison recomputes every vector the file prints itself and checks the result and every status flag against the reference model; the file chose those vectors, so passing it proves little
  the second comparison removes the file's test section and replaces it with a driver we generate, fed with vectors from the reference model, so that the file has no say in what it is asked: every operand pair where k^n <= 256, otherwise a seeded sample together with every edge case
  both have to agree completely, because 24 implementations once passed the first while the driver for the second had silently failed to build, so nothing independent had ever run them
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
