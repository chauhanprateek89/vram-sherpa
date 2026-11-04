# ADR-0015: Counts and verdict lists come from one list of cards, sorted and cut on the server

Date: 2026-11-02 · Status: proposed · Requirements: R-014, R-016, R-019, R-023, R-085, R-086, R-087, R-089 · Decisions: Q-30, Q-31, Q-32, Q-35 (Gate 1, 2026-11-02) · Constitution: rules 11, 12 and 16

## Context

`_group_results_by_model` skips a model with no usable variant, and `_summary_counts` works out each model's open variant separately. That duplication has been harmless because `Won't fit` had no cards. R-023 now needs each count to equal its list's total across all pages, and R-086 gives `Won't fit` models cards. Two copies of the "which variant does the card open on" rule could drift. Gate 1 (Q-31) also gave `Won't fit` its own order, closest to fitting first, with the sort keys recorded in `docs/decisions.md`.

## Decision

1. **One rule for the open variant.** `_default_variant(variants)`, in `main.py`, takes a model's variants in Q4, Q5, Q8, FP16 order and applies R-016: the recommended variant if usable, else the first usable; with none usable, the recommended variant, else the first.
2. **All models become cards.** `_group_results_by_model(rows)` returns one `ModelCard` per model left after filters, with no skip. Filters still apply earlier, in `_variant_query`, so filters and context reach the counts, the headline, Top Picks and the lists alike (R-089).
3. **Counts come from the cards.** `_summary_counts(cards)` tallies `card.default_variant.classification`. Each count is the length of its list by construction, and the headline's N is `Fits + Tight` (R-014), no longer the number of cards.
4. **`_verdict_cards(cards, label)` filters and sorts:**
   - `Fits`, `Tight`: `(-params_b, model_name, model_id)`.
   - `Won't fit`: `(required_vram_gb - available_vram_gb of the open variant, -params_b, model_name, model_id)`, compared unrounded.
   - `model_name` compares by code point, case-sensitively, as Python's `str` does today (R-019).
   - `model_id` is a last key that R-019 does not name. All 95 catalog names are distinct, so it changes nothing today. `Model.name` has no unique constraint and `_variant_query` has no `ORDER BY`, so without it two models sharing a name could swap between requests, and across pages one could appear twice and the other not at all. `model_id` is unique, so it makes each page stable.
5. **Cut after sorting.** `paginate` gives the slice, and only that slice goes to the template as `model_cards` (R-087, Q-35). The gist passes an empty list. The counts, the headline and Top Picks do not read the slice, so they cannot change with the page (R-085).
6. **Compute on every request, with no cache.** Rows are computed for every model on each request, as they are today (Q-35). Rule 11 rules out storing the view, and rule 12 rules out relying on database state beyond the seed.

## Checked against the catalog

Running this key over catalog `2026-10-23` on a temporary SQLite database reproduces every worked example in R-086: page 1, page 4 and page 5 of `/results?vram_gb=24&context_tokens=4096&verdict=wont-fit`, page 2 of `/results?vram_gb=1&verdict=wont-fit`, and `Qwen3-30B-A3B` at `20.9 GB over` with `quant_bucket=Q8`. The reference query's gist takes about 76 ms through the test client today, and that response carries 752 KB of HTML with 71 cards. A list page now carries 5 or 10.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Sort and page in SQL | Verdicts come from `estimate_variant` in Python. The formula would have to be repeated in SQL, against rule 16's one source for the estimate. |
| Keep `_summary_counts` working from rows | A second copy of R-016, and no structural guarantee that a count equals its list. |
| Largest first for `Won't fit` | The Gate 1 proposal, which the user changed (Q-31). |
| Round overage before comparing | R-019 says unrounded. R-086's page 4 example depends on it. |
| Cache rows per query | Adds invalidation and state for a 76 ms computation. Not needed. |

## Consequences

- `estimation.py`, `/how-it-works` and `ESTIMATION_POLICY_VERSION` do not change (R-086, rule 16). This is ordering and selection only; the figures are the ones `estimate_variant` already returns.
- No existing test calls `_summary_counts` or `_group_results_by_model` directly. Tests that read cards from a page, such as `tests/test_reference_queries.py` and `tests/test_r072_acceptance.py`, move to `verdict` URLs.
- A future duplicate model name falls to `model_id`, not to query row order, which improves on the note in `docs/decisions.md` without contradicting R-019.
- `UI: yes` through the lists and counts it feeds; the mechanism itself has no screen.
