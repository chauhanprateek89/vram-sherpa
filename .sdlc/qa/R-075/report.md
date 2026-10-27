# QA report: R-075 How it works page

Result: **PASS**. All 9 criteria pass. `passes: true` set on T-005 (T-019 was already `true`; it still passes).

Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027e`), Chromium via Playwright, branch `sdlc/full-ui-overhaul` at 1ceca19. 390 px with `is_mobile` + touch and 1280 px, light and dark. Fallbacks used scratch SQLite apps on ports 8772 to 8774 (model missing, Q4 missing, unseeded); the Docker database was not touched and the scratch servers are stopped. Mockups: `docs/design/final/how-it-works-*` (4 PNGs).

Evidence in this folder: `how-it-works-<width>-<theme>.png` (4 live renders), `mockup-compare.txt`, `audit.json`, `consistency-checks.json`, `card-llama-q4-breakdown-*.png`, `fallback-checks.txt`, `fallback-*.png` (8), `unchanged-checks.txt`.

## Mockup comparison (constitution rule 10)

All 4 renders have exactly the mockup's pixel size (390 x 4735, 1280 x 2457) and 0.00% of pixels differ by more than 40 in any channel. Console errors and warnings: 0.

## Criteria

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Keeps the inputs, four R-009 formulas, R-010 reserve formula, three R-011 rules, R-037 limitations, constants 1.10, 1.0, 0.05, 0.75, 0.10, word for word | PASS | One `h1`, six `h2` (Inputs, Formula, Reserve buffer, Classification, Worked example, Limitations). The five formula strings and three rule strings are identical to the page at f2ad2d8; the limitations sentence is exact; the reserve examples `With 4 GB ... 0.75 GB. With 16 GB ... 1.6 GB.` present. `tests/test_r075_acceptance.py::test_page_keeps_formulas_rules_constants_limitations_and_one_h1_six_h2` |
| 2 | States a policy version equal to the footer's | PASS | h1 `How VRAM Sherpa estimates fit (Policy v0.1)`; footer `Estimation policy: v0.1` on the same live page (`audit.json`) |
| 3 | Formulas, rules and steps wrap at 390 px, no horizontal scroll | PASS | 390 px, light and dark: `scrollWidth` 390 = 390; 0 scroll regions; 0 elements past the viewport; no `nowrap` on any `code`; formulas wrap to 3 or 4 lines inside right edge 334 px; the four steps and their calculations wrap in full (`how-it-works-390-dark.png`). Text under 12 px: 0 |
| 4 | Plain-language explanation of the ten terms | PASS | Each of `params_b`, `bits_effective`, `kv_gb_per_1k_ctx`, `context_tokens`, `weights_gb`, `kv_cache_gb`, `runtime_overhead_gb`, `required_vram_gb`, `available_vram_gb`, `reserve_gb` has a `code.name` and a sentence of 5 or more words without the raw name (test), with the old source lines kept; page matches the mockup (0.00%) |
| 5 | Worked example names Llama 3.1 8B Instruct at Q4, 2048 tokens, with the inputs and four steps | PASS | Lead `Llama 3.1 8B Instruct at Q4, with a 2048-token context. See its model page.`; inputs `params_b 8, bits_effective 4.5, kv_gb_per_1k_ctx 0.48, context_tokens 2048`; steps `4.95 GB (8 x 4.5 / 8 x 1.10 = 4.95)`, `0.98 GB (0.48 x 2048 / 1000 = 0.98304)`, `1.40 GB (1.0 + 0.05 x 8 = 1.40)`, `7.33 GB (4.95 + 0.98304 + 1.40 = 7.33304)`. Link `/models/model_llama_31_8b_instruct?context_tokens=2048` |
| 6 | Example equals the card's `GB needed`, the card's breakdown and the model page's Q4 row | PASS | `/results?vram_gb=24&context_tokens=2048&family=llama`, Q4 pressed on the `Llama 3.1 8B Instruct` card: `7.33 GB needed`; breakdown opened by tap or click: `Model weights 4.95`, `Context memory (KV cache) 0.98`, `Runtime overhead 1.40`, `Total needed 7.33`, all 4 width/theme runs. `/models/model_llama_31_8b_instruct?vram_gb=24&context_tokens=2048` Q4 row: required `7.33`, margin 16.67, Fits. `consistency-checks.json`, `card-llama-q4-breakdown-*.png` |
| 7 | Catalog values other than the above are shown, so the example still equals the card | PASS | Test sets the model's kv to 0.5 and the Q4 bits to 5.5 in the test DB: page shows `0.5`, `5.5` and steps 6.05, 1.02, 1.40, 8.47 (computed independently), and the model page's Q4 row shows the same required VRAM |
| 8 | Model or Q4 missing, or catalog unseeded: HTTP 200, every other section, unnamed example (`an 8B model at Q4`) with the fixed inputs and four steps, no model name | PASS | Scratch apps, all three cases (`fallback-checks.txt`): HTTP 200, six `h2`, all formulas and limitations present, lead `An example for an 8B model at Q4, with a 2048-token context.`, inputs 8 / 4.5 / 0.48 / 2048, steps 4.95 / 0.98 / 1.40 / 7.33 GB, no `Llama`, no `Instruct`, no `/models/` link, no `See its model page`; the test also checks that none of the 95 seeded model names appears. 390 and 1280 px, light and dark: `scrollWidth` equals width, 0 console errors (`fallback-*.png`) |
| 9 | `estimation.py` and `ESTIMATION_POLICY_VERSION` unchanged against main f2ad2d8 | PASS | `git diff f2ad2d8 -- src/vramsherpa/estimation.py` is empty and the blob hash of the working file is `6eebae1b...`, equal to that of f2ad2d8; `ESTIMATION_POLICY_VERSION = "v0.1"` at line 26 of `main.py` in both (`unchanged-checks.txt`) |

## Notes (not failures)

- The inline link `See its model page` is 21 px tall at every width. It is running text, R-075 sets no tap-target rule for it and it matches the approved mockup exactly.

## Test added

`tests/test_r075_acceptance.py` (8 tests, passing; `ruff check` and `ruff format --check` clean). Full `pytest` green.
