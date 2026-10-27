# QA report: R-070 Phone layout on every page (round 1)

Result: **FAIL**. 6 of 7 criteria pass; one fails (clipped GPU label in the `Search GPU` field). `passes` NOT set on T-010, T-011, T-016, T-019, T-021.

Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027d`), Chromium via Playwright, 390 px viewport with `is_mobile` and touch emulation, light and dark. All 24 approved states, plus longest-name probes. The unseeded state and the no-sources detail state ran on scratch SQLite apps (ports 8101, 8102, started from the branch code, stopped afterwards is not needed; they are local scratch). Helper scripts are in the QA scratchpad, not the repo.

Evidence in this folder: `summary.txt` (one line per state and theme), `audit.json` (full per-state measurements), `extra-checks.json` (header taps, 1280 px text sizes, long-GPU cards), `<state>-390-<theme>.png` (48 mockup states plus 12 longest-name probes) and `crop-*.png`.

## Mockup comparison (constitution rule 10)

The 24 states at 390 px, light and dark, were compared with `docs/design/final/<name>-390-<theme>.png` (card lists trimmed to the mockup's count). Same height every time. Pixels differing by more than 40: 0.00% on 44 renders, 0.01% on 2 (`results-filters-open` light, `unseeded` dark), 0.18% on `gpu-suggest` (the highlighted suggestion). Console errors: 0 in every state.

## Criteria

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | No horizontal scroll: `scrollWidth` equals 390, both themes, breakdown open, longest model names, longest GPU label in the field and in the summary | PASS | `sw=390`, `body=390` and a pan to x=80 leaves `scrollX=0` in all 60 state/theme runs. Breakdown open on the `Mistral NeMo 12B Instruct (2407)` and `DeepSeek-Coder-V2-Lite-Instruct` cards on a 16 GB laptop GPU (`long-models-breakdown-open-*`, `crop-card-*`); long GPU label in the field and summary (`long-gpu-field-*`, `crop-long-gpu-field-and-summary-390-light.png`); both long models' detail pages with the long GPU selected |
| 2 | No clipped, overlapping or ellipsised text; names, GPU labels, URLs and formulas wrap | **FAIL** | `input#gpu_search` holding `NVIDIA GeForce RTX 4090 Laptop GPU (16.0 GB)` shows only `NVIDIA GeForce RTX 4090`; text is 384 px wide in a 226 px field (`scrollWidth 384`, `clientWidth 226`). It is a single-line input that cannot wrap and the `Clear` button sits beside it. Seen in `results-gpu-*` (GPU picked from the list), `long-gpu-field-*` and `long-gpu-selected-mistral-*`; screenshot `crop-long-gpu-field-and-summary-390-light.png`. Everything else passes: 0 `text-overflow: ellipsis`, 0 line clamps, 0 text rectangles outside the viewport or an `overflow` ancestor (the 7 hits on detail pages are the visually hidden table header, 1x1 px by design), 0 overlapping text. The model-detail GPU `<select>` wraps the long label onto two lines inside its box (`crop-long-detail-select-390-light.png`). The summary's `Hardware:` line, card names, source URLs and formulas wrap |
| 3 | 44 x 44 tap targets, none overlapping | PASS, with one flag | Up to 442 targets per page across all states: 0 below 44 x 44 for buttons, chips, quantisation buttons, context presets, checkbox and radio with their labels, header links, theme toggle, disclosure summaries, and the styled GPU suggestions (`li[role=option]`). Open suggestion list overlays the form beneath it (a popover); the options do not overlap each other. Flag: on `/models/<id>` the source-URL links (own list items, not running text, not in the criterion's list) are `inline-block` with `padding: 8px 0; margin: -8px 0`: a URL that fits on one line is 38.5 px tall (`mistral.ai/news/mistral-nemo`, `long-detail-mistral-*`), and the boxes of neighbouring links overlap by 10 px. Judged outside the listed categories, so not counted; the orchestrator may rule otherwise. The mockup has the same markup (0.00% pixel diff) |
| 4 | Text, number and select fields at least 16 px; no visible text under 12 px, any width | PASS | All visible fields compute 16 px (`fields<16=0` everywhere; `Min params`/`Max params` measured in `results-filters-open`; select on model detail 16 px). Text nodes, `::before`/`::after` content, placeholders and SVG text: 0 below 12 px at 390 px in every state and at 1280 px on home, results, empty state, model detail and how-it-works (`extra-checks.json`, `minText1280`) |
| 5 | Numeric keyboards on `Available VRAM (GB)`, `Min params (B)`, `Max params (B)`, `Manual VRAM (GB)` | PASS | All four are `type="number"` with `inputmode="decimal"` on `/`, `/results` and `/models/<id>` (`audit.json`, `fields`). Automated: `tests/test_r070_acceptance.py` (5 tests). A real phone keyboard cannot be shown by headless Chromium, so this is the markup check |
| 6 | `Estimator`, `How it works` and the theme toggle reachable from the header with at most one tap before each | PASS | In all 60 runs the brand, both nav links and the toggle are fully inside the first viewport at scroll 0, not covered, and 44 px high: no tap needed before any of them. Real touch taps (`touchscreen.tap`) on `/`, `/results`, a model page and `/how-it-works`: `How it works` lands on `/how-it-works`, `Estimator` and the logo on `/`, the toggle switches light to dark with `aria-pressed=true` (`extra-checks.json`, `headerTaps`) |
| 7 | Each approved 390 px state matches its mockup (rule 10) | PASS | See "Mockup comparison" above |

## Why criterion 2 fails

R-070 names the longest GPU label "in the GPU field and in the summary" and says GPU labels SHALL wrap inside their container. The summary does. The field does not, and no screen state wraps it: a long label is cut off at the right edge of the field once selected, so the user sees `NVIDIA GeForce RTX 4090` and cannot see `Laptop GPU (16.0 GB)` without scrolling inside the field. A fix needs a design decision (for example a wrapping auto-growing field, or moving `Clear` below the field at phone width). The approved `gpu-suggest` mockup has the same single-line field, so the mockup itself does not meet the criterion at the longest label.

## Test added

`tests/test_r070_acceptance.py` (numeric keyboards, 5 tests, passing; `ruff check` and `ruff format --check` clean).

---

# Round 2 (2026-10-27)

Result: **PASS**. All 7 criteria pass against R-070 as amended on 2026-10-27 (input-value exemption, 44 px Sources links). `passes: true` set on T-010, T-019 (T-011, T-016 and T-021 also, see the R-073 report for T-016). T-011 and T-021 were already true.

Tested against http://localhost:8000 (assets `v=20261027e`, commit cd57789 included), Chromium via Playwright, 390 px, `is_mobile` + touch, light and dark. Helper scripts live in the QA scratchpad, not the repo. Evidence in this folder: `round2-audit.json` (per-state measurements), `round2-sources-all-models.json` (every model page), `r2-*-390-<theme>.png` (full pages), `r2-crop-gpu-field-and-hardware-390-<theme>.png`, `r2-crop-hardware-line-390-light.png`, `r2-crop-sources-mistral-390-<theme>.png`.

| # | Criterion (as amended) | Round 2 | Evidence |
| --- | --- | --- | --- |
| 1 | No horizontal scroll (`scrollWidth` 390), both themes, breakdown open, longest names, longest GPU label | PASS (spot-check) | `sw=390`, `body=390`, pan to x=80 leaves `scrollX=0` on `/results` (long GPU and manual 24 GB) and three model pages (Mistral NeMo, DeepSeek-Coder, Qwen) in both themes. Breakdown open on the Mistral NeMo and DeepSeek-Coder cards at 390 px: `sw=390`, five values inside the card (R-073 report) |
| 2 | No clipped, overlapping or ellipsised static text; input values may scroll if the full label is in the summary `Hardware` line | **PASS** | `input#gpu_search` holds `NVIDIA GeForce RTX 4090 Laptop GPU (16.0 GB)`; the value is 384 px wide in a 226 px field (scrolls inside the field, permitted for input values). The summary `Hardware` line reads `Hardware: NVIDIA GeForce RTX 4090 Laptop GPU (16.0 GB)` in full: it wraps onto two lines, `scrollWidth 312` = `clientWidth 312`, `overflow: visible`, right edge 346.9 px of a 351 px box (`r2-crop-hardware-line-390-light.png`). Same result when the GPU arrives through `gpu_search=<full label>` as through `gpu_id`. Static text: 0 `text-overflow: ellipsis`, 0 text nodes outside the viewport or an `overflow` ancestor (the 7 hits on model pages are the visually hidden 1x1 px table header, as in round 1), 0 elements with hidden horizontal overflow |
| 3 | 44 x 44 tap targets, none overlapping (buttons, chips, quant buttons, presets, checkbox/radio with labels, header links, theme toggle, disclosures) | PASS (spot-check) | 0 targets below 44 x 44 and 0 overlapping pairs on `/results` (both states) and the three model pages, both themes (running-text links exempt per the criterion) |
| 4 | Each Sources link at least 44 px tall, neighbours not overlapping (new) | **PASS** | All 95 model pages at 390 px: 491 source links, minimum height 44 px, 0 neighbouring overlaps, `scrollWidth` 390 on every page (`round2-sources-all-models.json`). Mistral NeMo page: 7 links, min 44 px; a one-line URL (`https://mistral.ai/news/mistral-nemo`) is now 44 px, not 38.5 px. Visual: `r2-crop-sources-mistral-390-light.png` and `-dark.png` |
| 5 | Text/number/select fields at least 16 px; no visible text under 12 px | PASS (spot-check) | `fields<16=0`, `text<12=0` on all 10 state/theme runs |
| 6 | Numeric keyboards on the four number fields | PASS | `tests/test_r070_acceptance.py`, 5 tests passing (markup `type="number"` with `inputmode="decimal"`) |
| 7 | Header reachable with at most one tap | PASS (spot-check) | Brand, `Estimator`, `How it works` and the theme toggle are 44 px high and fully inside the first viewport (top 23 and 75 px) on `/results` and a model page; a touch tap on the toggle sets `aria-pressed=true` and a tap on `How it works` reaches `/how-it-works`. Header markup and CSS are unchanged since round 1. Console errors: 0 on every page |

Round 1 failure resolved by the requirement amendment (the field value may scroll because the full label shows in the `Hardware` line), not by a layout change. Round 1 flag on Sources links resolved by commit cd57789.
