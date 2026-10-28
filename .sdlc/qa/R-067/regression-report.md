# QA regression report R-067, R-069, R-071, R-078 (task T-026): PASS

App at http://localhost:8000. Full `pytest`: all pass (incl. test_r069/071/078 acceptance).
Scripts: .sdlc/qa/R-067/scripts/cmp.py (pixel diff), beh.py (behaviour). Renders: .sdlc/qa/R-067/regress/<name>-<width>-light.png

## Mockup pixel diff (light, cards trimmed to the mockup's count; changed pixels, threshold 40)
results-from-landing (reached by typing `12 GB` in the landing box, submit): 1280 0.00%, 390 0.00%, same height as mockup.
results-gpu, results-manual-vram, results-errors: 0.00% at 1280 and 390. Console errors: none.

## Behaviour (Playwright, 1280 and 390)
- /results filters closed on load; label `Filters (2)` / `Filters (0)`; click opens; no JS: opens, `Apply filters` present, form submits, 71 cards.
- Top Picks copy exact; no `ranked`/`ranking`; empty state `There are 95 models in Won't fit, hidden by default.`
- Messages once: `?vram_gb=abc` one role=alert; `?gpu_search=RTX` one role=status; in-place refresh with 24 clears the error.
- Suggestion list on /results: typing 4060 gives the four R-071 labels in order, combobox/aria-expanded/listbox/option roles, 48 px options, Escape closes and keeps text, ArrowDown+Enter selects gpu_id and empties VRAM.
- Clear is `a#clear-gpu href="/"`, loads `/`. `/results` with no hardware 302 to `/`; `/results/partials/content?vram_gb=24` 302 to `/results?vram_gb=24`.
- Labels `Search GPU`, `Available VRAM (GB)`, `Check compatibility`, verdict labels unchanged.
