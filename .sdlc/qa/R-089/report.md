# R-089 QA report (T-045, T-046, T-048)

Browser: Playwright (python, chromium-1247) against http://localhost:8000. Scripts in `.sdlc/qa/R-089/scripts/`.
`pytest tests/test_r089_*.py`: 89 passed.

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Filters and presets stay in form, no sort control | PASS | Form markup shows family/quant/min/max/recommended and 2k/4k/8k in `#search-form` on gist and list; no sort control found. |
| 2 | Filters apply before counting | PASS | `/results?vram_gb=8&context_tokens=2048&family=llama` shows 6 / 4 / 4, headline "10 models fit or are tight", Filters (1). |
| 3 | Verdict not in Filters (N) or chips | PASS | `verdict=tight` gives `Filters (0)`; with family=llama gives `Filters (1)` and a chip only for Family. |
| 4 | Changes keep verdict and page_size, page 1 | PASS | p3.py: tight ps10 p2 then llama, Q4, min/max params, recommended: URL keeps `verdict=tight&page_size=10`, no page. Chip click and context change same (p4.py). No-JS `Apply filters` (apply_view=tight:10) keeps both. Try Q4 / Lower context only render on the verdict-less gist, so there is no verdict to carry; hrefs carry verdict+page_size when present (`_view_items`). |
| 5 | Tight 4k, page 3, 2k | PASS | "Tight: showing 1-5 of 9 models", URL `...context_tokens=2048...&verdict=tight`, no page. Screenshots `3-tight-2k-page1-1440.png`, `-390.png`. |
| 6 | Count 0 shows gist and message, recovers | PASS | Tight 8k: "No models in Tight with these settings.", `verdict=tight` kept; back to 4k: "Tight: showing 1-5 of 13 models". Screenshots `4-tight-8k-empty-gist-*.png`. |
| 7 | No verdict stays on gist | PASS | gist then 8k stays "Choose Fits, Tight or Won't fit...", no verdict (JS and no JS). |
| 8 | Check compatibility / Enter gives gist, no verdict/page/page_size, with and without JS | FAIL (narrow) | Passes for button and for Enter in VRAM and in typed-then-Enter in Min params, JS and no JS. Fails when keyup of the last character in Min/Max params arrives after the Enter submit: see Probe. |
| 9 | Clear loads `/` | PASS | url `/` (JS and no JS). |
| 10 | Back then change uses the shown view | PASS | Fits, Back to gist, 8k stays gist; tight 8k, Back to tight 4k, 2k gives Tight page 1 of 9; wont-fit/Back/Forward/Back to gist then 2k stays gist. |
| Probe | VRAM Enter with key still down, response over 350 ms | PASS for VRAM | p5.py: `#vram_gb` is not an auto-filter field; with a 1500 ms and a 100 ms delay and the key held 0 or 500 ms, the address ends `...vram_gb=6...` with no verdict and the gist is shown. |
| Probe variant | Same on Min params (B) / Max params (B) | FAIL | p5.py, `#min_params_b`: key down, Enter, key up. Two requests: first (submit) has no verdict, second (keyup, `js-auto-filter-text`, "changed delay:350ms") carries `verdict=tight&page_size=10`. Final URL `...min_params_b=6&max_params_b=&verdict=tight&page_size=10`, list shown instead of gist. Same with no hold. Typing then Enter 100 ms later (key already up) is fine (p6.py). |
| Mockup | gist-empty-all-inert, 1280 and 390, light and dark | PASS with note | The mockup query `min_params_b=5&max_params_b=1` is invalid, so the app adds the input-error alert (107 px at 390), absent from the mockup (`pixdiff.txt`, 6-27% diff). With a valid zero-result query (`min=500&max=600`, same height, `pixdiff-validrange.txt`) layout matches: heights equal (922 / 1472), diff 1.6-1.9% at 1280, 6-13% at 390 (background gradient and chip text). Visual check of 390 dark: identical layout. |
| Console | Errors | PASS | No console errors in any run. |

## Result

Tasks T-045, T-046, T-048 not marked passed: the Enter-submits-gist criterion fails in the keyup-after-submit race on Min/Max params fields.
Suggested fix: when a submit happens, cancel or ignore the pending `keyup ... delay:350ms` trigger (for example an `hx-sync` that drops, or checking a flag set on submit).

## Re-test after T-052

- `pytest tests/test_r089_*.py`: 100 passed.
- p6.py: min_params 6, Enter 100 ms later: gist, no verdict/page/page_size.
- p5.py: `#vram_gb` and `#min_params_b`, response 100 and 1500 ms, key held 0 and 500 ms: gist, no verdict, one request only.
- scripts/p8.py: `#vram_gb`, `#min_params_b`, `#max_params_b` x 100/1500 ms, last key's keyup after Enter: gist, no verdict/page/page_size, `Loading models` gone, no `aria-busy`, no `Could not load the models`, no console errors.
- Typing in Min/Max without Enter from page 2: refreshes after delay keeping `verdict=tight&page_size=10`, page dropped; a further edit refreshes again. Edit after a landed submit refreshes (stays on gist).
- Criterion 8 now PASS. All criteria pass. T-045, T-046, T-048, T-052 marked passes: true.
