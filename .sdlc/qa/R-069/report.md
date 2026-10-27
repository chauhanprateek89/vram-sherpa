# QA report: R-069

Result: **PASS** (13 of 13 criteria). Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027d`) in Chromium via Playwright, at 390 and 1280 px, in light and dark. Positions are the bottom edge plus scrollY, in CSS px.

The QA agent could not write this file itself (its Write call was refused), so the orchestrator recorded its returned report here. Evidence is in this folder: `positions.json`, `filters-interaction.json`, `mockup-compare.txt`, and 48+ captures, which are kept local and not committed. Test added: `tests/test_r069_acceptance.py` (24 cases).

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Headline first, then chips, Top Picks and cards | PASS | At 1280 on the reference query: answer tile at 92, Top Picks at 358, first card at 615 |
| 2 | Reference query at 1280 within 800 | PASS | Headline ends at 167; verdict at 770, margin at 769 |
| 3 | Message states at 1280 within 900 | PASS | Ambiguous 843, not found 798, input errors 883 |
| 4 | Every state with cards within 1688 at 390 | PASS | Verdict ends between 1243 and 1533 across 8 states. Headline ends at 804 |
| 5 | Filters closed at both widths, with or without JS, and no script opens them | PASS | `open === false` on load in every state and theme. No `open` attribute in the HTML |
| 6 | `Filters (N)` equals the chip count | PASS | Shows 0, 2, 3 and 4, each matching the chips shown |
| 7 | Opening the filters shows every R-027 to R-030 control, with and without JS | PASS | 7 family boxes, 4 quantisation boxes, min and max params, recommended only. `Apply filters` with JS off |
| 8 | Verdict and margin are the largest text in the pane | PASS | 17 px against at most 15 px for other text |
| 9 | Nothing added to a card beyond R-015 to R-021 | PASS | Text matches the approved mockups |
| 10 | Summary content kept | PASS | Hardware and context lines, 71 / 0 / 24 counts, the 8 GB note, R-003 wording, removal chips, 3 picks (none at 2 GB) |
| 11 | Empty state says models, never variants; singular for 1 | PASS | "There are 95 models…" and "There is 1 model…" |
| 12 | Top Picks show the approved descriptions | PASS | All three new lines; no "ranked" or "ranking" |
| 13 | Top Picks placement matches the mockup | PASS | Under the answer tile; open at 1280, closed at 390 (a 60 px row) |

## Mockup comparison

48 live-against-render comparisons. Page heights match exactly. 0.00% of pixels differ by more than 40, except results-filters-open (0.02–0.06%, anti-aliasing). No sideways scroll.

## Console

No errors. `vram_gb=abc` logs the browser's number-parse warning, because the rejected value is echoed back into the field (R-008, pre-existing).

## Note

With JS off at 1280, the first verdict ends at 745, because Top Picks stays closed without the script.
