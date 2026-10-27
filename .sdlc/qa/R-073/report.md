# QA report: R-073 "How this is estimated" breakdown

Result: **PASS**. All 5 criteria pass. `passes: true` set on T-016.

Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027e`), Chromium via Playwright. 390 px with `is_mobile` + touch (real taps) and 1280 px, light and dark, with `prefers-reduced-motion: reduce` and without. Mockups: `docs/design/final/why-closed-*`, `why-loading-*`, `why-loaded-*` (12 PNGs). Helper scripts are in the QA scratchpad, not the repo.

Evidence in this folder: `<state>-<width>-<theme>.png` (12 live renders, card list trimmed to the mockup's 4 cards), `mockup-compare.txt`, `functional-checks.json` (102 checks), `loading-checks.json` (26 checks), `open-*.png`, `loading-state-390-light-motion.png`, `long-mistral-open-390-light.png`, `long-deepseek-open-390-light.png`.

## Mockup comparison (constitution rule 10)

All 12 live renders have exactly the mockup's pixel size. Pixels differing by more than 40 in any channel: 0.000% on 8 renders, 0.032% (`loaded-390-dark`), 0.196% (`loading-390-light`), 0.243% (`loaded-390-light`). Remaining differences are small and sit in the tapped summary (touch/focus state), not in labels, rows or layout. Console errors and warnings: 0 in every run.

## Criteria

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Collapsed until opened, fetched once, shows `<quant label> on <model name>` with weights, KV cache, runtime overhead, reserve and required VRAM, each to 2 decimals, computed as R-020 | PASS | In all 8 width/theme/motion runs: 0 panels open and 0 `/results/why/` requests on load; one request after open, close, open again (`.../variant_model_deepseek_coder_v2_lite_instruct_q4?context_tokens=4096&vram_gb=24`). Title `Q4 on DeepSeek-Coder-V2-Lite-Instruct`; figures `9.72 / 3.85 / 1.79 / 15.36 / 2.40 GB`, all match `\d+\.\d{2} GB`. Static HTML: 284 `Open to load the breakdown.` placeholders for 284 panels, `hx-trigger="click once"`, `hx-swap="innerHTML"`. Figures recomputed from `data/seed_*.json` with the R-020 formula for 5 variants and hardware/context combinations by `tests/test_r073_acceptance.py` (5 tests passing) |
| 2 | Five values labelled with the Gate 2 wording; five figures unchanged | PASS | Labels in order: `Model weights`, `Context memory (KV cache)`, `Runtime overhead`, `Total needed` (class `total`, ruled off), `Safety reserve` (class `reserve`, last, with `Kept free, not part of the total`). None of the raw labels (`weights_gb`, `kv_cache_gb`, `runtime_overhead_gb`, `reserve_gb`, `required_vram_gb`) in the DOM. Total is before Safety reserve in DOM order. Mockup match 0.00 to 0.24% (above). Figures equal the formula (criterion 1) and `Total needed` = weights + KV + overhead within rounding on all 96 variants checked |
| 3 | Required VRAM in the breakdown equals the `GB needed` caption of the same pane | PASS | `15.36` equals `15.36` in all 8 runs; over 12 cards x 4 quantisations, tapped on a 24 GB manual run and on the 16 GB RTX 4090 Laptop GPU run (96 variants), `Total needed` equals the pane's `GB needed` in all 96, and the three terms sum to the total within 0.015. Also covered by existing `test_total_needed_equals_the_gb_needed_caption_of_the_same_pane` |
| 4 | Visible loading indication, visibly different from the loaded breakdown | PASS | With the request held: summary has `htmx-request`, `.why-loading` is `display: block`, `.why-idle` hidden, text `Loading the breakdown`, five bars (298 px wide at 390, 554 px at 1280, 14 px high), no `.why-list`. With motion: bars animate `shimmer 1.4s`. With reduced motion: animation `0s`, bars stay as flat tinted bars (light `rgba(40,30,110,.10)`, dark `rgba(255,255,255,.08)`) so the state is still visible. After the response the loaded list replaces the bars and `htmx-request` clears. Screenshots `loading-*` and `loaded-*` show the difference; loading renders match `why-loading-*` |
| 5 | Open at phone width: all five values within the card's width, no horizontal scroll | PASS | 390 px: `scrollWidth` 390; card right edge 363, widest value right edge 344, list `scrollWidth` = `clientWidth` (298). Held for 96 variants over 24 cards and on the `Mistral NeMo 12B Instruct (2407)` and `DeepSeek-Coder-V2-Lite-Instruct` cards on the 16 GB laptop GPU (`long-mistral-open-390-light.png`, `long-deepseek-open-390-light.png`). 1280 px: values inside the card too (right 612 of 633) |

## Test added

`tests/test_r073_acceptance.py` (5 tests, passing; `ruff check` and `ruff format --check` clean; full `pytest` run green).
