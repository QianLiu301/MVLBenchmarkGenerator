# Writing an introduction as a story line

Abstracted from R. Drechsler's story line for the LLM-MVL paper (ISMVL 2026), and used for
the MVL Benchmark Library paper in `story.md`.

A story line is written **before any prose**. Each line is one claim; in the final paper each
line becomes one to three sentences. If a line cannot be disputed by a reviewer, or cannot
carry a citation, it is decoration — delete it.

---

## The nine slots

| # | Slot | What the line does | Typical phrasing |
|---|---|---|---|
| 1 | Relevance | why this kind of work matters at all | *X is highly relevant for the evaluation and comparison of ...* |
| 2 | Narrowing | rule out the cheap alternative, name what actually counts | *... is not sufficient, but ... should be considered* |
| 3 | The established case | the neighbouring field where this is solved, with its classics | *since the early 80s ... were proposed \cite{}, later improved \cite{}* |
| 4 | Outgrowing (optional, strong) | the resource was used beyond its original purpose | *originally proposed for ..., they were later used intensively for ... as well* |
| 5 | Corroboration | other domains have it too | *also in other domains, like ..., the availability of ... \cite{}* |
| 6 | **The gap** | it is missing here — and the usual workaround does not count | *for ... very few are available; often ... has been used, but this does not reflect ... \cite{}* |
| 7 | **The closest attempt** | the one work a reviewer will raise, and its limitation | *in \cite{} an approach was proposed based on ...; this allows ..., but ... only* |
| 8 | The enabler | the new means, and its recent uses in the field | *recently ... have become available and used in ..., for an overview see \cite{}* |
| 9 | **The claim** | one sentence, one thing | *in this paper we show / present ...* |

Resource, tool and dataset papers add a tenth line: what the resource currently contains
(size, coverage, what is recorded). Method papers put no numbers in the introduction.

## The seven rules

1. **One claim per line.** Not one topic per line. If a line needs a semicolon and an "and",
   it is two lines.
2. **Citations hang on claims, never in a block.** With slots 3, 5, 6, 7, 8 cited, a separate
   Related Work section becomes unnecessary — worth half a page in a six-page paper.
3. **Build the gap in two stages**: absence (6), then the closest attempt and why it falls
   short (7). Slot 7 answers "but hasn't X already done this?" before the reviewer asks it.
4. **The enabler comes after the gap.** Reversed, the paper reads as chasing a fashion;
   in this order it reads as an answer.
5. **Exactly one "in this paper" line, and it is a claim, not a list.** Contributions are a
   bullet list later in Section I, not part of the story line.
6. **Do not polish.** Fragments, missing verbs and rough grammar are fine, and are how the
   original is written. Compression is what matters.
7. **10 to 14 lines.** Longer means slots are being padded.

## Self-check before sending

- Does every line move the argument one step? Cross out any line and see if anything is lost.
- Is the turn visible? Somewhere around slot 6 there must be a *but* / *however* /
  *very few are available*. That word is where the paper starts existing.
- Does the last line state one thing, or three?
- Would slot 7 survive the reviewer who wrote that cited paper?

## Where the template does not fit

- A paper whose contribution is a **reframing** rather than filling an agreed gap: the template
  will flatten it into "one more increment".
- A **negative or cautionary result**: there is no slot for consequences. Add one line after
  the gap saying what goes wrong if nothing changes.
- A **follow-on paper to one's own earlier work**: it needs two turns, not one — the original
  gap (now partly filled by the earlier paper) and the new one. This is the case in
  `story.md`, where slot 6 is the MVL benchmark gap and a second turn, "a generated design is
  not yet a benchmark", follows slot 8.

## Beyond the introduction

The same discipline extends to the whole paper: one line per claim, grouped under plain
section names. Send the introduction story line first; a professor or co-author judges whether
the paper should exist from those lines alone. Send the per-section lines only after the
premise is agreed.
