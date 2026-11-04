# R-084 QA (T-039, T-041, T-042)
All criteria PASS. Evidence: curl on http://localhost:8000, screenshots in this folder (1280/390, light/dark), full `pytest` exit 0 (incl. tests/test_r084_gist.py).
- Gist 200, document order (banner, Hardware, Context, controls, note, Top Picks; form with Filters (0) before): PASS (byte offsets 16507<16691<16745<16829<18271<18419).
- No card, with and without HX-Request: banner 1, `Manual VRAM (24 GB)` 1, `data-model-card` 0, `How this is estimated` 0: PASS.
- Top Picks `<details class="picks pane lite" open>` at both widths (checked in browser): PASS.
- Note `Choose Fits, Tight or Won't fit...` present, `Only Fits and Tight` absent: PASS.
- `/results?vram_gb=1`: empty state, `There are 95 models in Won't fit...`, Won't fit a link: PASS.
- Unseeded: covered by test_unseeded_catalog_shows_r026_and_the_message (pass).
- summary-banner present in ambiguous, not-found, invalid responses: PASS.
- POST /results gives 405; no new routes: PASS.
- Asset version `20261103i` (not 20261028f): PASS.
- Mockups: sizes identical except message-notfound 390 (1949 vs 1971 px; notice wraps in 2 lines instead of 3, minor). Layout, content and order match on visual comparison (reference 390 dark, notfound 390 light). No console errors; scrollWidth equals viewport at 1280 and 390.
Screenshots: .sdlc/qa/R-084/<name>-<width>-<scheme>.png
