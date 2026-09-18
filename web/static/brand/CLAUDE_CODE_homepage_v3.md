# Task: Homepage v3 (hybrid layout) + logo integration

Do not change routes, the database schema, or the review pipeline logic. This task covers only the homepage, the global header/footer, the logo, and the three data-consistency fixes in section 0. Keep the existing design tokens (--bg #FFFBEB, amber gradient #D97706→#F59E0B, teal --link #198C8A / --link-hover #0F766E).

---

## 0. Fix these first (correctness)

1. **Spec count inconsistency.** The previous version showed `alu_k3_8t` in the library, but the homepage now shows ALU = 0 specs. Investigate and report the cause, e.g. a status filter (only `approved`), records migrated without a status, or data lost during migration. Existing verified entries must be migrated to `status=approved` with their verification records intact. Add a test: the homepage category count equals the count of the Library page with the same filter.
2. **BibTeX author list is wrong.** It currently lists only "Rolf Drechsler". Load the citation from a single config file (`config/citation.bib` or equivalent) used by the homepage, the Cite page and the footer. Leave a TODO for the maintainer to confirm the final author list and title. Do not invent fields.
3. **BibTeX field.** Replace `note = {https://llm-mvl.com}` with `url = {https://llm-mvl.com}`.

## 1. Logo

Assets are in `mvl-logo/` (all text is outlined to paths, so there is no font dependency):

| File | Use |
|---|---|
| `mvl-logo-horizontal.svg` | Header (height 36px desktop, 30px mobile) |
| `mvl-logo-horizontal-dark.svg` | Dark backgrounds only |
| `mvl-logo-stacked.svg` | About page and README |
| `mvl-mark.svg` | Mark only (≥ 32px) |
| `mvl-favicon.svg`, `favicon-16.png`, `favicon-32.png`, `favicon-180.png` | `<link rel="icon" type="image/svg+xml">`, PNG fallbacks, `apple-touch-icon` (180) |

- Replace the current "layers" icon and the text-rendered title in the header with `mvl-logo-horizontal.svg`, wrapped in a link to `/` with `aria-label="MVL Benchmark Library, home"`.
- Do not recolor, stretch, or add shadows to the logo.
- Also copy the files into `static/brand/` and add a short `static/brand/README.md` with the usage table above.

## 2. Header

- Items: Library · Submit · Generate · About, then the Sign-in menu. Remove **Cite** from the main navigation; it stays in the footer and has its own page.
- Active item: 2px teal underline. Header background white with a 1px bottom border `#F3E8C8`; sticky, height 64px.

## 3. Homepage layout (top to bottom)

```
┌───────────────────────────────────────────────────────────┐
│ HERO (left-aligned, max-width 1100px)                     │
│  H1  An online resource for multi-valued logic benchmarks │
│  P   one-sentence subtitle (existing)                     │
│  [ search ...................................... ]        │
│  [Browse library]  [Submit a benchmark]  Download by selection │
├───────────────────────────────────────────────────────────┤
│ ABOUT (2 sentences, link "More about the library")        │
├───────────────────────────────────────────────────────────┤
│ BENCHMARKS                                                │
│  chips: All (n) | ALU (n) | Register file (n) | Processor (n) │
│  table: 10 latest approved specs      View all benchmarks │
├───────────────────────────────┬───────────────────────────┤
│ NEWS (3 latest items)         │ CITE (one line + Copy BibTeX) │
├───────────────────────────────┴───────────────────────────┤
│ FOOTER                                                    │
└───────────────────────────────────────────────────────────┘
```

### 3.1 Hero
- H1 reduced from the current size to `clamp(2rem, 4vw, 2.75rem)`, weight 700, left-aligned, with no single-word color accent (the logo already carries the teal).
- Subtitle: keep the current sentence, max-width 64ch, color --text-muted.
- Search: submits to `/library?q=...`; placeholder "Search by name, radix or structure, e.g. alu_k3, GF(5)".
- Buttons: primary "Browse library" (gradient), secondary "Submit a benchmark" (outlined), plus a text link "Download by selection", which opens `/library?mode=download` (see 3.4).
- Vertical padding: 56px top, 40px bottom. The hero must not exceed 60% of a 1366×768 viewport.

### 3.2 About (replaces the long Welcome block)
Exactly this text; move the full three-paragraph version to `/about`:

> The MVL Benchmark Library collects specifications and reference implementations of multi-valued logic designs — starting with arithmetic-logic units over GF(p), GF(pⁿ) and Z/kZ — in C, Python, Verilog and VHDL. Each implementation is checked against an independent golden model and published with its verification record. [More about the library](/about)

Render it as plain text on the page background, not inside a card.

### 3.3 Benchmarks (replaces the three category cards and "Recently added")
- **Chips** (single-select filter): All, ALU, Register file, Processor. Each shows its count of **approved specs**. A chip with count 0 is rendered disabled (muted, `aria-disabled="true"`, tooltip "No published benchmarks yet") but stays in the row.
  - Rename "RISC-V CPU" to **Processor** everywhere in the UI. Keep the internal enum value if renaming it would require a migration, and add a TODO noting that the MVL processor encoding must be defined before publishing entries.
- **Table**: the 10 most recently approved specs, filtered by the selected chip (client-side, or via `?module=`).
  Columns: Name (monospace, link to detail page) | Module | k | Digits | Structure | Operations (chips, max 4, then "+N") | Implementations (one badge per language; verified = teal, pending = amber, absent = hidden) | Added (relative date, full ISO date in `title`).
  - Right-align numeric columns and use `font-variant-numeric: tabular-nums`.
  - Legend below the table: "Teal badge: verified against the golden model · Amber badge: pending review". Remove the old "C" example badge.
  - Link below the table on the right: "View all benchmarks", pointing to `/library`.
  - Tablet and mobile: the table scrolls horizontally inside its own container; below 640px it renders as a stacked list (name + k/digits/structure line + badges).
- **Empty state** (0 approved specs overall): a single compact row inside the table area — "No benchmarks are published yet. [Submit the first one] or [generate one]." Max height 120px. Never render a half-page empty box.

### 3.4 Download by selection (MQT-Bench-style)
On `/library?mode=download`, show a selection panel above the table:
- Multi-select for module, k, digits, structure and language; checkbox "Verified implementations only" (checked by default).
- Live counter: "N specs, M files, ≈ X MB".
- Button "Download selection (.zip)". The server builds the zip containing `manifest.json` (filters used, spec IDs, SHA-256 per file, format version, generation timestamp) and `CITATION.bib`.
- Limit: at most 500 specs per request; show an error message if exceeded. Cache zips keyed by a hash of the normalized filter set.

### 3.5 News
- New table `news(id, date, title, body_md, link)` with admin CRUD (reuse the existing admin auth).
- Homepage shows the 3 latest items as date + title (+ optional link); no cards.
- Seed items (the maintainer will edit them; mark them as drafts, not published):
  - "Benchmark format v1.0 released"
  - "Library opened for submissions"

### 3.6 Cite (compact)
- One line: "If you use the library, please cite our ISMVL 2026 paper." plus a "Copy BibTeX" button (copies from `config/citation.bib`; show a toast "BibTeX copied") and a link "Citation details", pointing to `/cite`.
- On `/cite`, show the BibTeX in a `<pre>` with `white-space: pre-wrap`. There must be no horizontal scrollbar.

## 4. Footer
Line 1: "Rolf Drechsler, Qian Liu — Group of Computer Architecture (AGRA), University of Bremen" (names link as now).
Line 2 links: Format v1.0 · Acknowledgements · Cite · License (CC BY 4.0) · GitHub · Imprint / Legal notice.
Everything must fit on two lines on desktop; wrap cleanly on mobile.

## 5. Style rules for this task
- Remove dashed borders used as decoration (they read as "disabled"). Use a 1px solid `#F3E8C8` border only where a container is needed.
- Use no more than one card-style container per homepage section.
- Use sentence case for headings, buttons and chips; no ALL-CAPS labels; no "→" appended to link text.
- Icons: lucide only, 18px, stroke 1.75.
- Accessibility: visible focus rings (2px teal outline, 2px offset); honor `prefers-reduced-motion`; contrast ≥ 4.5:1 for body text (use #0F766E for small teal text on #FFFBEB).

## 6. Acceptance checklist (report each item as pass/fail)
- [ ] Homepage category counts equal the Library counts for the same filter (automated test).
- [ ] The BibTeX shown on the homepage, on `/cite` and in the download zip is byte-identical.
- [ ] No horizontal scrollbar on the homepage at 375px, 768px or 1366px widths.
- [ ] At 1366×768 the benchmark table header is visible without scrolling.
- [ ] Favicon appears in the browser tab; apple-touch-icon is present.
- [ ] Empty state renders compactly when the database has zero approved specs (test with an empty fixture).
- [ ] Lighthouse accessibility score ≥ 95.
- [ ] Provide before/after screenshots at 1366px and 375px.
