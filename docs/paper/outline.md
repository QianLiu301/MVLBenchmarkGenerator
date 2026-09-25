# Paper outline — ISMVL 2027

Venue: ISMVL 2027, Kongsberg, Norway, 1–3 June 2027.
**Submission deadline: 1 November 2026.** 6 pages, IEEE two-column, 50–100 word abstract.
Topics to submit under: *Computer Arithmetic* / *Verification*; a special session on
3VL, ternary and mixed-radix computing is announced.

---

## Thesis, in one line

**Benchmarks that carry their own evidence.** What an entry computes is written down, whether
it is correct has been checked by an oracle independent of whoever produced it, and the check
is published so anyone can repeat it — you do not have to trust us.

## Story line

A candidate — from a generator or from a person — becomes a citable benchmark in three beats:

1. **say what it must compute** (specification),
2. **let it be checked** (oracle and evidence),
3. **then you can put it to work** (uses).

The library is the subject; one entry (`alu_k3_8t`) is the vehicle that carries the reader
through the three beats; the collection frames it before and after.

## Title

**The MVL Benchmark Library: Multi-Valued Arithmetic Benchmarks with Published Verification Records**

Alternatives: *Benchmarks That Carry Their Own Evidence: The MVL Benchmark Library* ·
*The MVL Benchmark Library: An Online Resource for Multi-Valued Arithmetic Benchmarks*

## Abstract (draft, 92 words)

> Multi-valued logic has no common set of high-level benchmarks. We present the MVL Benchmark
> Library (llm-mvl.com), a collection of arithmetic-logic specifications over Z/kⁿZ and
> GF(q)[x]/(xⁿ), each with implementations in C, Python, Verilog and VHDL. An entry states the
> algebra it computes in and what its flags mean, and carries a verification record: the vectors
> compared against an independent reference model, the strength of that comparison, and the tools
> used. Release 1.1 contains N specifications and M implementations, is archived under a DOI, and
> accepts submissions under the same rules.

## Contributions (the bullet list at the end of §I)

1. **A name does not define a multi-valued benchmark.** An "8-trit ALU over GF(3)" may denote
   Z/3⁸Z, GF(3⁸) or GF(3)[x]/(x⁸) — three different algebras — and "negative" has no agreed
   meaning without two's complement. We define a format in which the algebra, every operation and
   every flag is part of the entry.
2. **An oracle independent of the producer — and validated itself.** The reference model is derived
   from those definitions and uses no generator, no library entry and no simulator. It is checked
   against arbitrary-precision integer arithmetic, an independent Galois-field library, the ring and
   field axioms, and the binary special case. This found a defect in our own polynomial reduction.
3. **Evidence a candidate cannot anticipate.** Every implementation is compared twice: on the vectors
   it prints itself, and on injected vectors after its test section is replaced — exhaustively where
   the operand space allows. Each verdict is published with tool versions, vector counts, seed and
   checksum.
4. **The resource.** N specifications spanning k = 2…9 and 8…14 digits in two algebraic families,
   M implementations in four languages, browsable, downloadable, queryable, DOI-archived, and open to
   submissions. Implementations that fail are published with their reports, because which producer
   fails on which specification is part of what the library records.

---

## Section and subsection headings

Reading only the headings should give the argument:

```
I.   Introduction
II.  The Library at a Glance
     A. A parameterised family, not a fixed list
     B. What it contains
     C. How to obtain and cite it
III. What an Entry States                     <- beat 1
     A. Naming and parameters
     B. Family M: the ring Z/k^n Z
     C. Family F: the ring GF(q)[x]/(x^n)
     D. Why the algebra belongs in the entry
IV.  What an Entry Proves                     <- beat 2
     A. An oracle independent of the producer
     B. Checking the checker
     C. Testimony and cross-examination
     D. What the verdict records
     E. What the verdict does not claim
V.   Using the Library                        <- beat 3
     A. As a device under test
     B. As a software reference
     C. Varying the radix with everything else fixed
     D. Comparing descriptions of one specification
     E. Evaluating producers of hardware descriptions
     F. What this library is not
VI.  Conclusion and Roadmap
```

Two headings are deliberately statements rather than labels: *Checking the checker*
says at a glance that the oracle is not taken on trust, and *What the verdict does not
claim* / *What this library is not* draw the scope before a reviewer has to ask.

---

## Sections

### I. Introduction — 1.0 p
- Benchmarks decide how EDA methods are compared. Binary logic has ISCAS/ITC/EPFL; multi-valued
  logic has almost nothing, and what exists is either encoded binary or randomly generated from
  decision diagrams.
- [1] showed that designs can now be generated at negligible cost, and listed what a complete suite
  would still need: more widths, more radices, more logics, more languages, more abstraction levels.
- Supply is therefore no longer the problem. **Two others take its place.** A name does not determine
  what a design computes. And a generated design arrives with a testbench that it passes.
- The delta in one sentence: *[1] demonstrated that one such design can be generated; this paper is
  about what has to hold before a generated design can be used as a benchmark by someone else.*
- Contributions (4 bullets above). Structure of the paper.
- Related work folded in: ISCAS/ITC [Brglez], RevLib [Wille et al. 2008], MDD-based MVL generation
  [Radmanovic & Stankovic 2018], SystemC modelling [Grosse et al. 2003], LLMs in EDA [Jha et al.].

### II. The Library at a Glance — 0.75 p
- **A. A parameterised family, not a fixed list.** Module type x radix k x digit count n x algebraic
  family. Adding a radix or a width adds entries without changing anything else, which is what [1]
  asked for in its Sections IV.A and IV.B.
- **B. What it contains.** N specifications over k = 2..9 and n = 8..14, M implementations in C,
  Python, Verilog and VHDL, V verified; failures kept with their reports.
- **C. How to obtain and cite it.** Browse and filter; per-entry, filtered or whole-library archives;
  JSON API; DOI-archived release; CC BY 4.0. Longevity: fixed snapshots, a public submission and
  review process, maintained at the institution.
- *Table I: coverage matrix - radix down, digits across, verified languages per cell.*

### III. What an Entry States — 1.0 p
The running example (`alu_k3_8t`) enters here and stays to the end of Section IV.
- **A. Naming and parameters.** Stable names derived from the parameters; the scheme extends to
  register files and processors.
- **B. Family M: the ring Z/k^n Z.** Radix-k integers; the six operations with their formulas; carry
  and borrow; Z and N defined explicitly.
- **C. Family F: the ring GF(q)[x]/(x^n).** Digit-wise field arithmetic; multiplication as the
  polynomial product truncated to n digits; the irreducible polynomial fixed per q; no carry, N = 0.
- **D. Why the algebra belongs in the entry.** Three algebras share one informal name, and
  multi-valued logic has no flag convention comparable to two's complement, so entries in different
  papers that share a name are not otherwise comparable.
- *Listing 1: the specification of the running example.*

### IV. What an Entry Proves — 1.25 p
- **A. An oracle independent of the producer.** Derived from the definitions of Section III; uses no
  generator, no library entry and no simulator; versioned, and every record names the version.
- **B. Checking the checker.** Four cross-checks from outside the project: arbitrary-precision
  integers; an independent Galois-field library, table by table, under the same irreducible
  polynomial; the ring and field axioms, including SUB o ADD = id and NEG = SUB(0, .); and k = 2
  reproducing an ordinary binary ALU. This found a real defect - polynomial reduction used the
  irreducible polynomial in reverse order, which affects GF(8) and GF(16). The library now contains
  k = 8, so an unvalidated oracle would have produced wrong verdicts.
- **C. Testimony and cross-examination.** Every vector the file prints is recomputed and compared,
  result and all flags; then the file's test section is replaced by a driver fed with oracle vectors -
  all operand pairs where k^n <= 256, otherwise a seeded sample plus every edge case. Both must agree
  completely. Why testimony alone proves nothing: 24 implementations once passed it while their
  injected harness had silently failed to build.
- **D. What the verdict records.** Tool versions, vectors compared and passed per strategy, the
  strength label, the file checksum, the oracle version, the simulator log - enough to repeat it.
- **E. What the verdict does not claim.** Simulation against an oracle, not a formal proof; exhaustive
  only where the operand space is small. Raising the injected sample from 50 to 5000 operand pairs
  changed no verdict, so sample size is not the limiting factor. Entries whose operand range exceeds
  the harness's integer range are marked not applicable rather than checked.
- *Fig. 1: the check in one picture. Listing 2: a verification record.*

### V. Using the Library — 1.25 p
- **A. As a device under test.** Every HDL entry exposes the same ports and opcode encoding, so one
  testbench drives any of them. *Listing 3: the interface.*
- **B. As a software reference.** The C and Python versions of a specification compute the same
  function and are checked by the same oracle.
- **C. Varying the radix with everything else fixed.** k = 2..9 with the same operations and widths -
  a controlled axis that hand-built benchmarks rarely afford.
- **D. Comparing descriptions of one specification.** Four descriptions judged by one oracle, so any
  two agree on the checked vectors by construction.
- **E. Evaluating producers of hardware descriptions.** Failed candidates are kept with their reports
  and provenance. Software descriptions are verified almost always; the hardware description
  languages far less often - the same specification, the same producer. *Tables II and III.*
- **F. What this library is not.** High-level functional benchmarks, not structurally optimised
  netlists; they fix what is computed, not how. Evaluating physical synthesis quality needs a
  different kind of benchmark.

### VI. Conclusion and Roadmap — 0.25 p
A submission carries a manifest and its files, runs through the same checks, and receives a
maintainer decision within a published time. Roadmap: true extension fields GF(q^n) for the GF(2^m)
algorithms cited in [1], register files and processors, SystemC, gate-level netlists.

### References — 0.5 p
~12 entries.

---

## Figures, listings and tables (6 + 1 figure)

| # | What | Section |
|---|---|---|
| Table I | Coverage matrix: radix × digits, verified languages per cell | II |
| Listing 1 | An entry's specification | III |
| Fig. 1 | The checking pipeline in one picture | IV |
| Listing 2 | A verification record | IV |
| Listing 3 | The HDL interface for using an entry as a DUT | V |
| Table II | Verified rate by language | V |
| Table III | Verified rate by radix and algebraic family | V |

## Deliberately excluded

- Prompts and generation dialogue — that is [1].
- A per-model comparison table — model coverage is unbalanced and one sample per cell; only
  aggregate and per-language figures are reported.
- A synthesis experiment (e.g. area versus radix) — it would demonstrate a use but pulls the paper
  off its line; left to future work.
- Website screenshots.

## Open items before writing

- Final numbers after the parameter-grid expansion (k = 8, 9 and widths 9, 11, 13) and the
  sensitivity run.
- Literature check for other multi-valued benchmark efforts, beyond those cited by [1].
- Author list and its relation to [1].
- Publish the prompt template used for generation, for reproducibility.
