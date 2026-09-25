# Literature census: how MVL work is evaluated today

Collected 2026-09-25 to support the "the field is active but cannot compare itself"
paragraph of the introduction (the RevLib move). Each item is marked with how firmly it is
established, because these claims will carry citations.

## Bottom line

No multiple-valued equivalent of RevLib, SATLIB or ISCAS exists. Searching explicitly for one
returns RevLib, SATLIB and TSPLIB themselves, and — already — our own repository. The gap the
paper claims is real and nobody has filled it.

What exists instead are four separate evaluation practices, none of them shared across
communities, and none of them consisting of designs at the level our library provides.

## A. Synthesis and minimisation: 50,000 randomly generated functions

**Established.** The de-facto benchmark of the direct-cover / heuristic MVL synthesis
literature is *50,000 randomly generated 2-variable 4-valued functions*, sometimes with a
second set of 50,000 2-variable 5-valued functions. It appears from at least the 2010
ant-colony work through 2024.

Confirmed in detail for: *Swarm intelligence versus direct cover algorithms in synthesis of
Multi-Valued Logic functions*, Applied Computing and Informatics 20(1-2), 2024 (open access).
It generates the 50,000 functions itself, with a stated minterm distribution (500 functions
with 16 minterms, 2,700 with 15, down to 75 with 6), **gives no seed, and does not make the
set available**. It then tabulates its averages against ARM, Besslich, Dueck-Miller and
Espresso-MV — numbers published by other authors who each generated their own 50,000
functions.

Same benchmark in: *Improved direct cover heuristic algorithms for synthesis of MVL
functions*, Int. J. Electronics 101(2), 2014; *Ant colony optimisation–direct cover*, Int. J.
Electronics 97(12), 2010; *A New Heuristic Tool for Designing MVL Systems*, J. Circuits,
Systems and Computers, 2023.

Why this matters for us: the sets are **random, tiny (2 variables), regenerated per paper,
unseeded and undistributed** — and cross-paper tables are built from them anyway. This is
RevLib's paragraph 3, but sharper: RevLib complained that circuits were not published and cost
metrics differed; here the benchmark itself is not the same benchmark from paper to paper.
It is also exactly what the professor's own line rules out — random functions rather than
structures occurring in designs.

## B. Decision diagrams: binary benchmarks with grouped variables

**Established.** MDD work is evaluated on ISCAS-85 and MCNC-91 *binary* circuits, with binary
variables partitioned into groups and treated as multi-valued. This is current practice, not
only the 1999 criticism we already cite (\cite{BK:1999}).

## C. Device-level ternary and quaternary design: no shared designs at all

**Established as a body of work; the incomparability claim is well supported but the strongest
single quote is not yet pinned (see open items).** Very active in 2024-2026: CNTFET, GNRFET,
memristor and TDDFET ternary cells in Scientific Reports, IEEE TCAD, Science Advances and
several ScienceDirect journals. Each paper designs its own STI / NTI / PTI, half adder and
multiplier and reports SPICE power-delay product under its own device model, supply voltage,
temperature and frequency; comparison is against numbers copied from other papers' tables.

A 2024 critical review exists (*Design implementations of ternary logic systems: A critical
review*, ScienceDirect) but is paywalled; worth getting through the university for a direct
statement about comparability.

## D. Ternary architecture: benchmarked against binary, and a different digit convention

**Established, and the most useful item we found.** At ISMVL 2025 (Montreal, 41 papers in 10
sessions): *REBEL-6: A 32-Trit Balanced Ternary Instruction Set Architecture with R2R Compiler
Pipeline for C*, S. Bos, V. Bodahl, O. C. Moholth, H. Gundersen. A 32-trit ISA, with an open
compiler pipeline, **benchmarked against the binary RV32I** — because no ternary reference
exists to benchmark against. Reported: 1.4% fewer instruction executions, 33.2% lower dynamic
power.

Two things follow.

1. It is a direct instance of the workaround that a quoted claim in the literature describes:
   in the absence of MVL benchmarks, binary benchmarks are used.
2. REBEL-6 is **balanced** ternary (digits -1, 0, 1); our library is unbalanced (digits
   0 … k-1) with modular or field semantics. Two serious ternary ALU efforts, same word
   "ternary", different digit sets. **This is our naming-ambiguity argument occurring in the
   field, not as a hypothetical.** It should be used descriptively — conventions differ, which
   is why a library has to state its own — never as a criticism.

The authors are the Ternary Research Group at the University of South-Eastern Norway,
Kongsberg, founded 2019. **ISMVL 2027 is in Kongsberg.** They are very likely hosts, and are
plausibly reviewers. The paper should treat balanced ternary as a legitimate convention the
format could be extended to, and say so in future work.

## What this gives the introduction

Three lines that we could not write before:

- the field is active — ISMVL 2025 alone had 41 papers across synthesis, decision diagrams,
  architecture, emerging devices and security
- but each community evaluates on its own material: randomly generated small functions in
  synthesis, encoded binary circuits for decision diagrams, self-designed cells at device
  level, and a binary ISA as the reference for a ternary architecture
- and the material is usually not distributed, so published comparisons rest on sets that are
  not the same set

Plus a sharper version of the naming line, now with a real example: balanced versus unbalanced
ternary.

## Open items

- **Pin the quotable sentence.** A sentence to the effect of "since there are no standard
  benchmark functions available for comparing MVL designs, benchmark functions for binary
  logic design are often used" appears in search results but could not be traced to a specific
  paper. It is the single most useful citation available to us — someone else stating the gap.
  Candidates: *Performance evaluation of multiple-valued logic circuits using a statistical
  approach* (IEEE, ~2005); *Representation of Multiple-Valued Logic Functions* (Stankovic,
  Astola, Moraga). **Do not cite it until the source is confirmed from the actual paper.**
- Get the 2024 ternary critical review through the university library.
- dblp and IEEE Xplore were not reachable from here; ISMVL 2024 and 2026 have not been
  censused. ISMVL 2025 was obtained from the conference program page.
- Check whether MVSIS (Berkeley) ships a benchmark set, and whether it is still available.
