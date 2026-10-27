# QA report: R-072 (model cards, quantisation switcher and fit gauge)

Result: **PASS** (5 of 5 criteria). Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027d`) in Chromium via Playwright: 390x844 with touch and real taps, 1280x800, light and dark, JavaScript on and off. Console checked with Chrome DevTools and Playwright: no errors on any page.

Pages: results-gpu (RTX 4090 Laptop 16 GB, 4096), results-manual-vram (24 GB, 4096), results-tight-gated (8 GB, 2048, Llama 8B), plus 8 GB and 6 GB lists with many Won't fit variants. 71 cards on the 16 GB and 24 GB pages.

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Cards, default quantisation, switcher, gauge, captions, ordering keep R-015 to R-019 and R-021, incl. gauge `role="img"` label and server-rendered widths | PASS | `cards-audit.json`: 5 pages x 4 viewport/theme runs, 0 problems. Per card: name link `/models/<id>?...` with context and hardware; `<n>B params` and family; buttons ordered Q4,Q5,Q8,FP16; exactly one `aria-pressed="true"`, never a Won't fit default when a usable variant exists; one visible pane matching it; Won't fit buttons have `data-oversize`, 2 px line-through and `Does not fit in N GB`; `<m> GB spare/over`; bar width = min(100, required/available x 100) and reserve width = reserve/available x 100 (within 0.6 pt of the label's rounded figures); captions `<req 2dp> GB needed` and `<avail 1dp> GB`; label `Needs X of Y gigabytes, with Z gigabytes held in reserve`; cards ordered by classification, params desc, name asc; `Gated` tag with the licence tooltip (21 gated cards). Switcher (`interaction-audit.json`, `switch`): tapping Q5, Q8, FP16 shows that pane, updates `aria-pressed`, makes 0 network requests; Enter on a focused button works, 3 px focus outline. JS off: inline widths 91.7% and 10.0% render at 91.7 and 10.0% of the track. `pytest tests/test_reference_queries.py`: 5 passed (figures and order); full `pytest` passes. |
| 2 | All four quantisation buttons on one row at phone width | PASS | At 390 px every card (35 to 71 per page, light and dark) has 4 buttons with the same `top` (e.g. 1211), each 73x46 px, inside the `.quants` box and inside the card. No horizontal scroll. Also one row with JS off. At 1280 px: 52x46 px, one row. `card-results-tight-gated-390-*.png` |
| 3 | `Fits`, `Tight`, `Won't fit` differ by label plus a non-colour cue, in both themes | PASS | Label text plus shape: circle with tick, triangle with bar, rounded square with cross. Ink-mask difference between pairs: Fits/Won't 17 to 18%, others 49 to 54%, all four runs. Greyscale renders (`gray-*.png`, CSS `grayscale(1)`) show all three distinct. `verdict-*.png` |
| 4 | Reserve band visible and distinct from the bar, both themes | PASS | `.resv` is a 2 px dashed outline 22 px tall around a 14 px track (4 px proud top and bottom), contrast against the track 8.6:1 light, 11.7:1 dark. Where the bar overlaps it (Tight, Won't fit) the outline still shows above and below the bar and cannot be confused with a fill. `gauge-tight-*.png`, `gray-*.png`. Note: colour contrast of the outline against the bar itself is only 1.1 to 2.0:1 (`reserve` in the JSON), so distinctness there rests on the dashed outline and its overhang, not on colour. |
| 5 | A `Won't fit` button distinguishable from pressed and unpressed buttons without colour | PASS | Computed style: Won't fit buttons (pressed or not) have `text-decoration: line-through` 2 px; Fits buttons have `none`, pressed and unpressed, in 4 runs. The pressed Won't fit button is struck and shows the pressed fill. `card-wont-pressed-*.png` |

## Mockup comparison (rule 10)

`mockup-compare.txt`: results-gpu, results-manual-vram and results-tight-gated, 390 and 1280, light and dark. results-tight-gated is the same size as the mockup (390x2748, 1280x1277) with 0.0000% of pixels differing by more than 40 at all four. The gpu and manual-vram mockups are samples of the 71-card list, so the top 1500 px (390) and 1000 px (1280) were compared: 0.0000% differ, except 0.0009% (manual-vram 390 dark) and 0.08% (gpu and manual-vram 1280 dark), which are sub-pixel differences. No missing element, layout or content difference.

## Notes (not failures)

1. The requirement's Source line mentions a "hatched" reserve band; the built band is a dashed outline. The criterion (visible and distinct) holds and it matches the mockups.
2. The reserve outline is low-contrast against an overlapping bar (note on row 4); it relies on shape.
3. Live assets are `v=20261027d`; the repo has `20261027e` (CSS-only model-page link fix). Cards are not affected.

## Evidence and tests

Folder `.sdlc/qa/R-072/`: `cards-audit.json`, `interaction-audit.json`, `mockup-compare.txt`, screenshots `results-*`, `card-*`, `verdict-*`, `gray-*`, `gauge-tight-*`, `nojs-*` at 390 and 1280, light and dark.
Test added: `tests/test_r072_acceptance.py` (5 cases: button order, pressed and gauge counts, tooltip and strike rule, four-column grid, three distinct verdict shapes, dashed reserve outline). `ruff check` and `pytest` pass.
