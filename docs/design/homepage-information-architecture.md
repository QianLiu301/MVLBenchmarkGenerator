# Homepage and information architecture — analysis and proposal

Status: **agreed** (2026-09-22); decisions in section 6, implementation follows.
Scope: the homepage, the top navigation and the footer of llm-mvl.com.

---

## 1. Who the homepage is for

An academic benchmark library has three kinds of visitor, and they arrive with
different questions. The homepage has to answer all three within one screen.

| Visitor | Question | What they need first |
|---|---|---|
| **Returning user with a task** ("I need the k = 5 VHDL files") | Where is the data? | Search, and drill-down by parameter with counts |
| **First-time evaluator** (referee, PhD student, tool author) | Can I trust this, and what is in it? | Size of the collection, how entries are verified, license, how to cite |
| **Contributor** | Can I add to it, and what will be checked? | Submit, the format, the review rules |

Today the site serves the first reasonably well, the second and third badly —
the pages that answer them exist but are reachable only from the footer.

---

## 2. What comparable resources actually do

Four sites, read on 2026-09-22.

**SuiteSparse Matrix Collection** (sparse.tamu.edu) — the homepage *is* the
collection: one sentence of identity, a filter panel, then the table of all
2904 matrices with a sort control and paging. No hero, no cards. The only
navigation link is "about".

**UCI Machine Learning Repository** (archive.ics.uci.edu) — one welcome line
that contains the size ("We currently maintain 689 datasets"), two buttons
(*View datasets*, *Contribute a dataset*), then *Popular datasets* and
*New datasets* as short lists with one-line descriptions.

**OEIS** (oeis.org) — the search box is the page. Under it, a single compact
row of utility links (Lookup · Welcome · Wiki · Register · Contribute · Format ·
Style Sheet · Transforms · Recents). The footer carries only who maintains it,
the last-modified date, the total count and the license.

**RevLib** (revlib.org) — the closest analogue, from the same group. Structure:
1. "Welcome to RevLib!" plus one paragraph saying what it is, what it provides and
   that researchers may submit;
2. a pointer line to *learn more* / *documentation* / *acknowledgements*;
3. a notice line about the current file-format version;
4. **two drill-down columns with counts** — functions by category
   (ALUs (6), Arithmetic Functions (44) …) ending in *All functions (154)*, and
   circuits by library ending in *All circuits (454)*;
5. three cards at the bottom: **Documentation**, **RevKit**, **Submit**, each one
   sentence plus "more".

### The common pattern

1. Identity in **one** short paragraph, and it contains the size of the collection.
2. **Entry into the data above the fold** — search and/or categories with counts.
   No long marketing hero.
3. A small number of explicit "what you can do next" paths (use · docs · contribute).
4. Counts everywhere: for an academic resource the numbers are the credibility signal.
5. The footer carries **meta only** — maintainer, license, last modified, legal.
   Documentation is never hidden there.

---

## 3. What is wrong with our homepage today

1. **The identity is written twice.** The hero subtitle and the "about-line"
   paragraph say nearly the same thing (~60 words), which pushes the data below
   the fold for no gain.
2. **Our strongest claim is unsupported on the page.** "Checked against an
   independent golden model" is asserted in both paragraphs, but /format,
   /review-process and the reference-model section — the pages that *prove* it,
   and the main thing we have that RevLib does not — appear only in the footer.
3. **The footer is a dumping ground**: nine links mixing documentation
   (Getting started, API, Format, Review process, Models) with meta (Cite,
   License, Acknowledgements, Imprint). A link in a footer is close to unpublished.
4. **The top navigation is under-used**: four items, while five documentation
   pages have no home in the navigation at all.
5. **No numbers above the fold.** 24 specifications / 216 implementations /
   140 verified / 5 generation models is the fastest trust signal we have, and it
   currently appears only as small counts inside chips, halfway down.
6. **"Generate" sits at the same level as "Library" and "Submit"**, although it is
   password-gated and is the project's old purpose, not the library's.
7. **The Cite block takes a full column** at the bottom and duplicates /cite.
8. **Nothing says who it is for.** A synthesis or verification researcher cannot
   tell in five seconds what they would do with these files.

---

## 4. Proposed information architecture

### 4.1 Top navigation (every page)

```
Library   Submit   Docs ▾   About                       [Sign in ▾]
                   └ Getting started
                     Benchmark format v1.0
                     Review process
                     Generation models
                     API
```

- **Generate** moves into the signed-in menu. It is password-gated already, so it
  does not belong in the public navigation.
- Everything a reader needs to judge or use the library is now at most two clicks
  from any page.

### 4.2 Footer (meta only, two lines)

```
Rolf Drechsler, Qian Liu
Cite · License (CC BY 4.0) · Acknowledgements · Imprint / Legal notice
Release 1.0 (2026-09-22) · 24 specifications · 216 implementations ·
format v1.0 · reference model v1.1
```

The status line is the OEIS pattern ("Last modified … Contains 399469 sequences"):
it dates the site on every page and costs one row.

### 4.3 Homepage, section by section

Revised after comparing with OpenCores and RevLib: the identity is a titled panel
rather than a paragraph, and the entry into the data is a pair of boxed category
panels rather than a row of loose chips.

| # | Section | Content | Model |
|---|---|---|---|
| 1 | **Hero** (compact) | Title, one sentence, search box, two buttons: *Browse library*, *Submit a benchmark* | - |
| 2 | **"What is the MVL Benchmark Library?"** panel, left, with an **"At a glance"** box on the right | See 4.3.1 and 4.3.2 | OpenCores |
| 3 | **Category panels**: *Specifications* and *Implementations* | See 4.3.3 | RevLib |
| 4 | **Latest benchmarks** | Existing table + *View all* | UCI "New datasets" |
| 5 | **Three paths**: Use it, How it is verified, Contribute | Cards with one sentence and a link | RevLib's Documentation / RevKit / Submit |
| 6 | **News + release line** | Three news items; *Release 1.0 - doi:... - Copy BibTeX* | OEIS footer status |

Removed: the long about-line paragraph (replaced by the panel), the full-width Cite
block (becomes one line), *Download by selection* in the hero (it belongs on the
Library page, next to the filters).

#### 4.3.1 The "What is ...?" panel

OpenCores' block is a titled bar, a bold one-line positioning statement, three short
paragraphs and a closing invitation. Ours, same shape:

> **What is the MVL Benchmark Library?**
>
> **A reference collection of verified benchmarks for multi-valued logic design.**
>
> The library collects specifications of multi-valued arithmetic — currently
> arithmetic-logic units over the ring Z/kⁿZ (radix-k integer arithmetic with carry)
> and over GF(q)[x]/(xⁿ) (digit-wise Galois-field arithmetic) — together with
> implementations of each specification in C, Python, Verilog and VHDL.
>
> Every implementation is compiled or simulated and compared, vector by vector, with
> an independent reference model, and is published with the verification record that
> produced it: tool versions, vector count, seed and file checksum. Nothing is
> published as correct because a model or an author said so.
> *(links: how it is verified · benchmark format)*
>
> The files are meant for anyone who needs realistic non-binary designs: authors of
> synthesis and verification tools, researchers comparing multi-valued arithmetic,
> and anyone measuring how well code generators handle non-binary hardware.
>
> Contributions are welcome — a submission runs through the same pipeline and a
> maintainer decides within 14 days. *(link: submit a benchmark)*

This also fixes the "nothing says who it is for" problem from section 3.8.

#### 4.3.2 The "At a glance" box (right of the panel)

```
Specifications        24
Implementations      216
Verified             140  (65 %)
Languages              4
Generation models      5
Release 1.0 · 2026-09-22 · doi:10.5281/zenodo.22895517
```

#### 4.3.3 The two category panels

RevLib's homepage has one panel per object type — *Functions (Categories)* and
*Circuit Realizations* — each a list of links with counts, closing with an
"All ... (n)" link. Our two object types map exactly: **specifications** and
**implementations**.

```
+- Specifications ---------------+   +- Implementations --------------+
| Module                         |   | Language                       |
|   ALU                    (24)  |   |   C                      (25)  |
|   Register file      (planned) |   |   Python                 (24)  |
|   Processor          (planned) |   |   Verilog                (85)  |
|                                |   |   VHDL                   (82)  |
| Structure                      |   |                                |
|   Z/k^n Z                (20)  |   | Verification                   |
|   GF(q)[x]/(x^n)          (4)  |   |   Verified              (140)  |
|                                |   |   Did not pass           (76)  |
| Radix k                        |   |                                |
|   2 (4)  3 (4)  4 (4)          |   | Source                         |
|   5 (4)  6 (4)  7 (4)          |   |   LLM-generated         (215)  |
|                                |   |   Human-authored          (1)  |
| Digits                         |   |                                |
|   8 (6) 10 (6) 12 (6) 14 (6)   |   |                                |
|                                |   |                                |
| -> All specifications    (24)  |   | -> All implementations  (216)  |
+--------------------------------+   +--------------------------------+
```

Every entry links into the browse filters that already work
(`/library?k_value=3`, `/library?language=vhdl`, `/library?verified=1`, ...).
Two notes:

- **"Did not pass (76)"** needs a new filter value (`verified=0`); today only
  `verified=1` exists. A small addition to `_apply_filters`.
- **By generation model** would be a natural fourth group in the right panel
  (deepseek-flash 59, gpt-5 59, codestral 48, gemini 39, gpt-oss 7), but there is no
  `model` filter yet. Either add one, or link that group heading to /models.

This panel pair replaces the "Browse by parameter" chip row added earlier the same
day: same idea in the boxed form the reference sites use, and it carries three axes
the chips did not (module, verification status, source).

### 4.4 The two points raised directly

- *"Getting started should sit on the same line as Compare generation models"* —
  solved by removing that paragraph altogether: those links become navigation
  (Docs) and the three path cards, where they are far more visible.
- *"The footer has too many links; should the others be on the homepage instead?"* —
  yes. Documentation goes to the Docs menu **and** the three cards; the footer keeps
  only meta. This is what all four reference sites do.

---

## 5. Further suggestions

Beyond the layout, in rough order of value for effort:

1. **Label the empty categories as planned, and say so once.** "Register file" and
   "Processor" showing (0) reads as broken; "(planned)" plus one line — *register
   files and processors are the next module types; a specification proposal is
   welcome* — turns a gap into a roadmap and an invitation.
2. **A statistics page** (`/statistics`): specifications per radix and digit count,
   implementations per language, verification rate per language and per model,
   growth over time. The data is already in the database; SuiteSparse and OEIS both
   have one, and it is the natural home for the deferred failure-mode analysis.
3. **"Recently added implementations"** as a dated list, separate from News (News is
   announcements, this is activity), optionally with an Atom feed. OEIS has
   "Recents"; researchers do subscribe to feeds.
4. **Named download bundles**: one click for *all verified VHDL*, *all C*, *the k = 3
   family*, *everything*. Today a visitor has to build a selection first; these are
   pre-filled links to the selection endpoint that already exists.
5. **An "MVL primer" page**: what multi-valued logic is, what radix-k arithmetic with
   carry means, how Z/kⁿZ differs from GF(q)[x]/(xⁿ), and why both are in the library.
   Format section 2 is precise but written for someone who already knows; a primer
   serves newcomers and search engines, and gives the panel text something to link to.
6. **Make the search box understand the obvious shorthands** — `k=3`, `vhdl`,
   `GF(4)`, `alu_k3` — and say so in the placeholder. The search box is the first
   thing a returning visitor touches.
7. **One icon per panel and card** (Lucide, already in use), the way OpenCores and
   RevLib mark their boxes: enough structure to scan, without decoration.

## 6. Decisions (2026-09-22)

1. **Docs**: a `/docs` index page listing each document with a one-line description,
   reachable from the navigation, plus a dropdown on desktop so the individual pages
   are one click away. The index degrades gracefully on mobile, where dropdowns are
   awkward.
2. **Generate** moves into the signed-in menu; it leaves the public navigation.
3. **News**: kept, but written as release notes, which is what comparable resources
   do — SuiteSparse has no news at all, UCI expresses "new" as *New datasets*, OEIS
   and RevLib each carry a single terse notice line about the current format or a
   call for contributions. Concretely: each entry is a date, one line and a link to
   the thing that changed; the homepage shows the three latest next to the release
   line, and `/news` holds the archive. "What is new in the data" stays the job of
   the *Latest benchmarks* table.
4. **Three path cards** confirmed: *Use it* · *How it is verified* · *Contribute*.
5. **Category panels**: two panels as drawn, and the right one gains a
   **by generation model** group, which requires a new `model` filter on the browse
   page (and `verified=0` for "did not pass").
6. **No source-code or tooling pointer** anywhere on the site.

## 7. Not in this proposal

The benchmark page, the browse page and the submission flow are unchanged. The
failure-mode analysis of the generation models stays deferred; it would extend
/models, not the homepage.
