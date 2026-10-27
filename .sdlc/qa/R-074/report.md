# QA report: R-074 Model detail page

Result: **PASS**. All 5 criteria pass. `passes: true` set on T-018.

Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027e`), Chromium via Playwright, branch `sdlc/full-ui-overhaul` at 1ceca19. 390 px with `is_mobile` + touch (real taps) and 1280 px, light and dark. The no-sources state used a scratch SQLite app (all `sources` cleared) on another port, since no real model lacks sources; it has been stopped. Mockups: `docs/design/final/detail-gated-*`, `detail-ungated-*`, `detail-no-sources-*`, `detail-error-*` (16 PNGs). Renders are taken with the viewport stretched to the page height (the backdrop is `position: fixed`, README line 152).

Evidence in this folder: `<state>-<width>-<theme>.png` (16 live renders), `audit.json` (per cell geometry, overflow, console), `mockup-compare.txt`, `text-compare.txt`, `behaviour-checks.json`.

## Mockup comparison (constitution rule 10)

- Text: visible text identical to the mockup in all 8 state/width pairs (`text-compare.txt`, 0 diff lines).
- `detail-no-sources`: 0.00% of pixels differ by more than 40 in any channel, in all 4 width/theme renders, same height.
- `detail-gated`, `detail-ungated`, `detail-error`: identical down to the first Sources link, then shifted. Heights are +120 px (390) and +123 px (1280; +80 on the error state) against the mockup. This is the agreed departure of 2026-10-27 (QA R-070 rows of `docs/decisions.md`): Sources and provenance links are now 44 px tall. Nothing else differs in text, order or layout.

## Criteria

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Model facts, KV estimate, sources, gated notice, recompute form and variant list keep the content and behaviour of R-033 to R-036; GPU dropdown, R-035 precedence and fields restyled only | PASS | Four fact tiles (Family qwen, Params 7.6B, Type instruct, License apache-2.0), `KV cache estimate per 1k context: 0.460 GB`, Sources links all `target=_blank rel="noopener noreferrer"`, 404 `{"detail":"Model not found"}`. Gated model (Llama 3.1 8B): `gated` tag and the R-036 notice; open model: neither. Form (390 touch and 1280, `behaviour-checks.json`): first option `Manual VRAM`, 36 options; picking a GPU empties VRAM; typing VRAM resets the GPU to Manual; GPU + vram_gb 12 gives available 16.00 (the GPU) and the help text; neither gives 8.00; blank gives 8.00 with no alert; `vram_gb=abc` gives 200 with `Invalid value for vram_gb.`, `role=alert`, `aria-invalid=true`; `inputmode=decimal`; `Recompute` is `.btn-primary`. Browser console: 0 errors on all states except the note below. `tests/test_r074_acceptance.py` (101 tests) covers the same over HTTP |
| 2 | Every variant shows label and bucket, recommended marker, bits, required, available, margin (2 decimals) and classification | PASS | Test recomputes each row from `data/seed_*.json` with the R-009 to R-011 formulas for all 95 models at 12 GB / 4096 plus 3 other hardware/context combinations: label `Q4 (Q4) *`, bits 1 decimal, required, available, margin 2 decimals, `Fits`/`Tight`/`Won't fit`, sorted FP16, Q4, Q5, Q8. All pass. Seven column headings kept in order; `data-label` on every cell (7 per row); `role` table/row/columnheader/cell; no `table-scroll` |
| 3 | At phone width each variant's label, required, available, margin and classification show with no horizontal scrolling, of the page or a region | PASS | 390 px, all 4 states, light and dark: `scrollWidth` 390 = `clientWidth` 390; 0 elements with `overflow-x: auto/scroll` that overflow; 0 elements whose right edge is past 390; in all 4 rows of each state every cell has left >= 0 and right <= 390. Interactive elements under 44 px: 0 (links are 62 to 65 px when they wrap to two lines). Screenshots e.g. `detail-ungated-390-light.png` |
| 4 | `/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096` at 390 px: model name and first variant's classification within the first 1688 px | PASS | H1 `Qwen2.5 7B Instruct` bottom at 202 px; the first variant (FP16) classification `Fits` ends at 1214 px (mockup 1174 px, +40 from the 44 px source links), limit 1688. Same in light and dark. Gated model: 1434 px (mockup 1394). Error state: 1305 px |
| 5 | Each variant's provenance (notes and sources) comes after that variant's figures and classification | PASS | At 390 each row's Provenance cell top is at or below the Status cell bottom in all 16 rows; at 1280 it is the last column of the same row. Markup order checked per row in `test_provenance_follows_the_figures_and_classification_in_every_row` |

## Notes (not failures)

- `detail-error` logs the Chrome warning `The specified value "abc" cannot be parsed, or is out of range.` (`type=number` given `value="abc"`). It is a warning, not an error; `main` at f2ad2d8 echoes `manual_vram_input` into the same `type="number"` input, so it is not caused by this change.
- The mockup's 1280 px source links are 38.5 px; the live page makes them 44 px (agreed departure).

## Test added

`tests/test_r074_acceptance.py` (101 tests, passing; `ruff check` and `ruff format --check` clean). Full `pytest` green, `ruff check .` and `ruff format --check .` clean.
