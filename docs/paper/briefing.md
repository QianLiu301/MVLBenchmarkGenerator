# Briefing: second ISMVL paper on the MVL Benchmark Library

Self-contained background for discussing what this paper should be. Everything here is
factual: numbers come from the live library, the summary of the prior paper comes from
its text.

Constraints: ISMVL, **6 pages max**, abstract **50–100 words**.

---

## 1. The prior paper this one extends

R. Drechsler, *"LLM-based Generation of High-Level Benchmarks for MVL Designs"*, ISMVL 2026
(single author, 6 pages). Referred to below as **[1]**.

**What it does.** It shows that an LLM can produce a meaningful MVL benchmark with very
little effort. A dialogue of four or five prompts with ChatGPT-5 turns "design an 8-bit
ALU, but 3-valued" into an 8-trit ALU over GF(3) *in integer style* (operands 0…6560,
arithmetic modulo 3⁸ = 6561), emitted as compilable C with a random testbench. The paper
prints the generated code, compiles it, and shows the output of 20 random test vectors.

**Two details that matter for us.**

- The dialogue itself shows the semantics being *chosen by the model and then picked by
  the human*: the model offers Kleene K3, Łukasiewicz L3, and {0,1,U} for verification; then
  offers "integer style" versus "vector style" (eight independent 1-trit ALUs). The
  benchmark's meaning is settled by an informal exchange, not by a written specification.
- Footnote 4 states plainly that because the design was converted from a two-valued ALU,
  "not all flags are supported… and corrections are needed for a complete design". The
  correctness of the artifact is acknowledged as incomplete.

**Its Section IV ("Discussion") is a roadmap** of what a *complete* benchmark suite would
additionally need:

| | Item in [1] | Status in our work |
|---|---|---|
| A | Datapath scalability: 8 → 16 → 32 bits, and fine steps such as 9, 10, 11 | Partly: 8, 10, 12, 14 digits for every k; no 16/32, no odd widths |
| B | Scalability in the number of logic values k; "families of functionally equivalent but differently valued circuits" | **Done**: k = 2…7, same operations, same four widths |
| C | Choice of logic: Łukasiewicz L3, Kleene, Galois fields; pointer to GF-based algorithm work [11], [12] | Partly: modular ring Z/kⁿZ and truncated polynomial ring GF(q)[x]/(xⁿ). No L3/Kleene, and **no true extension-field GF(kⁿ) word arithmetic**, which is what [12] (GF(2^m) verification) would need |
| D | Other description languages: Python, SystemC, Verilog, VHDL; for HLS, logic synthesis, formal verification | **Done** except SystemC, and all four exist *for the same specification* |
| E | Other abstraction levels: behavioural ↔ RTL ↔ gate-level netlist, for refinement and equivalence studies | Partly: software and RTL, no netlists. But every level is checked against one common oracle, so any two verified implementations of a specification agree on the checked vectors by construction |

**What the roadmap does not mention at all: correctness.** Every item is about coverage —
more widths, more k, more logics, more languages, more levels.

---

## 2. What we built (facts)

A public resource at llm-mvl.com, archived as release 1.0 on Zenodo
(doi:10.5281/zenodo.22895517).

- **24 specifications**: ALUs with ADD, SUB, MUL, NEG, INC, DEC over k ∈ {2,3,4,5,6,7} and
  n ∈ {8,10,12,14} digits.
- **Two algebraic families**, written out unambiguously per entry:
  - *modular*: the ring Z/kⁿZ — radix-k integer arithmetic, carry and borrow defined,
    Z and N flags defined (k = 2,3,5,6,7);
  - *field*: the ring GF(q)[x]/(xⁿ) — digit-wise Galois-field arithmetic, MUL is the
    polynomial product truncated to n digits, no carry between digits (k = 4).
- **217 implementations** in C, Python, Verilog and VHDL, produced by several models
  (deepseek-flash, gpt-5, codestral, gemini-2.5-flash, gpt-oss-120b) plus two written by hand.
- **141 verified (65 %)**. Failures are published with their reports, labelled as failures.
- Per implementation: tool versions, vector counts, seed, SHA-256, simulator log, and the
  reference-model version used.
- Format specification, review rules, JSON API, submission pipeline with maintainer review.

### The verification method (the part [1] has no counterpart for)

1. **An independent reference model.** The two families are implemented once, directly from
   their algebraic definitions. It uses no language model, no entry of the library and no
   simulator, so it is independent of everything it judges. It is versioned; every
   verification record names the version it was produced with.
2. **The reference model is itself validated** against four things that do not come from the
   project: the language's own arbitrary-precision integer arithmetic (exhaustively for
   small kⁿ, otherwise 20 000 seeded pairs); an independent Galois-field library, table by
   table, under the same irreducible polynomial; the ring and field axioms directly
   (associativity, distributivity, inverses, SUB∘ADD = id, NEG = SUB(0,·), INC = ADD(·,1));
   and the requirement that k = 2 reproduce an ordinary binary ALU. *This suite found a real
   defect*: polynomial reduction used the irreducible polynomial in reverse order, affecting
   non-palindromic polynomials (GF(8), GF(16)). Fixed in v1.1.
3. **Two comparison strategies per file.** (A) every vector the file prints itself is
   recomputed by the reference model; (B) the file's test section is replaced by a driver fed
   with reference vectors — exhaustive over all operand pairs when kⁿ ≤ 256, otherwise 50
   seeded random pairs plus edge cases. Both must pass.

Strategy B is not cosmetic. At one point 24 C implementations were passing on strategy A
while their injected-vector harness had silently failed to build, so nothing independent had
ever exercised them.

### What the data shows

| | verified / total | rate |
|---|---|---|
| C | 25 / 25 | 100 % |
| Python | 24 / 24 | 100 % |
| Verilog | 53 / 85 | 62 % |
| VHDL | 39 / 83 | 47 % |

| radix | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 |
|---|---|---|---|---|---|---|
| rate | 78 % | 67 % | **75 %** | 57 % | 50 % | 58 % |

| digits | 8 | 10 | 12 | 14 |
|---|---|---|---|---|
| rate | 67 % | 67 % | 62 % | 63 % |

Three observations:

1. **The same specification yields correct software and wrong hardware.** C and Python are
   essentially always right; the HDL versions of the same specification fail 38 % / 53 % of
   the time.
2. **Carry propagation, not exotic algebra, is what breaks.** k = 4 is the Galois-field
   family — digit-wise, no carry between digits — and scores 75 %, better than the
   carry-propagating k = 5, 6, 7 (57 %, 50 %, 58 %). Operand width barely matters at all.
3. **A single pass rate hides the failure mode.** One model fails mostly at compile time,
   another compiles and computes the wrong answer, a third fails at run time in VHDL only.

Caveat to state honestly in the paper: model coverage is unbalanced (59, 59, 48, 39, 7
implementations), one sample per (specification, language, model), one fixed prompt, no
temperature study. So per-model comparison is *not* claimed; only aggregate and per-language
figures are reported.

---

## 3. Proposed thesis

> Prompt-level generation does not produce a benchmark; it produces a **candidate**. Turning
> candidates into a suite needs two things the roadmap in [1] does not mention: a
> **machine-checkable specification of the intended semantics**, and an **oracle independent
> of the generator**. With both, generation scales across k, width and language exactly as [1]
> predicted — and roughly a third of what comes out is still wrong, invisibly so, because a
> generated file's own testbench passes.

This extends [1] rather than repeating it: [1] establishes that generation is cheap; this
paper establishes what it takes for the result to be usable, and delivers the resource.

### Candidate titles

- *The MVL Benchmark Library: An Online Resource for Verified Multi-Valued Benchmarks*
- *From Generated Candidates to a Verified MVL Benchmark Suite*
- *Trusting Generated Benchmarks: Specification and Verification for MVL Designs*

### Abstract draft (89 words)

> Research in multi-valued logic lacks a common set of high-level benchmarks: published
> designs are small, ad hoc, and their semantics are rarely stated precisely enough to
> reproduce. We present the MVL Benchmark Library, an online resource of arithmetic-logic
> specifications over Z/kⁿZ and GF(q)[x]/(xⁿ), each with implementations in C, Python,
> Verilog and VHDL. Every implementation is simulated and compared vector by vector against
> an independent reference model, and is published with the record that produced it. Release
> 1.0 contains 24 specifications and 217 implementations.

### Six-page outline

| Section | Pages |
|---|---|
| I Introduction: benchmarks in EDA, the MVL gap, what [1] showed, what remains | 0.75 |
| II Related resources: ISCAS/ITC, RevLib, MDD-based MVL generation, [1] | 0.5 |
| III Specifying a benchmark: naming, the two families, operation and flag semantics | 1.0 |
| IV The reference model and its own validation | 1.0 |
| V Verification: the two strategies, what a record contains | 1.0 |
| VI The library: content, access, submission, DOI | 0.75 |
| VII What the data shows: language gap, carry effect, failure classes | 0.75 |
| VIII Conclusion and roadmap: higher k, true extension fields, SystemC, netlists | 0.25 |

---

## 4. Open questions to discuss

1. **Headline**: resource paper (library first, findings as a section) or findings paper
   (the 65 % and the carry effect first, library as the artifact)? The data supports either;
   the resource framing needs no further experiments.
2. **How much of Section VII to include** given the unbalanced model coverage. Aggregate and
   per-language only, or also per-model with an explicit caveat?
3. **Terminology.** [1] calls arithmetic modulo 3⁸ "over GF(3)". Z/3⁸Z is not a field, and
   the distinction decides what an entry means; the library uses Z/kⁿZ versus GF(q)[x]/(xⁿ).
   The paper needs to introduce this notation. Frame it as terminology that a library must
   pin down, without correcting anyone.
4. **Scope of the roadmap section**: which of the remaining items (true GF(kⁿ) arithmetic for
   the algorithms cited in [1]; Łukasiewicz/Kleene logics; 16/32 digits; SystemC; gate-level
   netlists) to promise as future work.
5. Whether to include a figure of the pipeline and a coverage matrix, given six pages.
