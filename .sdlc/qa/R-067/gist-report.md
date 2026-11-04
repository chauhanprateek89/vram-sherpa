# R-067 (R-084 change): G1 to G13 and rows 2-10, 15-18, QA 2026-11-04

App: http://localhost:8000 (docker, branch sdlc/full-results-gist); unseeded state on a scratch SQLite app (port 8101, empty DB).
Method: Playwright Chromium, 390 and 1280, light and dark. Live screenshot vs `docs/design/final/gist-<name>-<w>-<theme>.png` (pixel diff, per-pixel channel delta > 40) and visible text vs the rendered mockup HTML.
Script: `.sdlc/qa/R-067/gist-scripts/capture_gist.py`; live screenshots: `.sdlc/qa/R-067/gist/<name>-<w>-<theme>.png`; rows 2-10/15-18: `.sdlc/qa/R-067/rows/`.

## Criterion 2: G1 to G13, 26 mockups x 4 renders = 104 comparisons: PASS
Heights equal the mockup (dH=0) and the pixel diff is at most 0.4% in all but the exceptions below. scrollWidth equals the viewport everywhere. First card verdict bottom 896/871 at 1280 (limit 900), 1471/1446 at 390 (limit 1688).
- G1 reference, G2 all-three, G8 verdict-zero, G9 messages (ambiguous, notfound, invalid), G13 focus-verdict: diff 0.08-0.14%, text identical.
- G3 list-fits, G4 middle, G5 last and size10, G6 tight / short / size10, G7 wont-fit / page10: diff <= 0.33%, card counts 5/5/1/10/5/4/9/5/5, text identical apart from a visually hidden `Results` heading (`h2.sr-only`) that the mockup does not draw.
- G10 empty-wont-link: 0.0%.
- G11 unseeded: 0.18-0.4%. Text differs in one chip: live `Example GPU` (empty DB has no GPU to name) vs mockup `RTX 4060 8GB`. Data-derived, accepted.
- G12 loading-from-gist, failed-from-gist, loading-next-page, failed-next-page (request to `/results/partials/content` held or aborted): `Loading models` status / `Could not load the models. Try again.` alert shown, pending ring on control, list and gist stay. diff 0.0-0.83% (390 light/dark), text identical.
- G13 focus-size, focus-page, focus-heading, focus-verdict: 3px outline on the right element (`Page 7`, `10`, `H2#list-heading`, Fits tile). focus-heading pixel diff is 11-17% only because my capture does not scroll to the mockup's crop; the live page after `Next` focuses `h2#list-heading` with a solid 3px outline (checked in script output).
- G10 empty-all-inert: NOTE, 1280 diff 1.3%, 390 height +107. The mockup's URL (`min_params_b=5&max_params_b=1`) is rejected by the app's own validation (`main.py:1179`), so the live page adds a `Please fix the following input issues` banner the mockup does not draw. That check predates this change. The layout (three inert tiles, chips, `Filters (2)`, empty state below) is as drawn. A valid query (`min_params_b=500&max_params_b=600`) gives the same look without the banner (`rows/empty-all-inert-validquery-*`). Treated as a mockup query artefact, not a build failure; raise it at review if a stricter reading is wanted.

## Criterion 1: rows 2-10 and 15-18 (change parts) : PASS
Form, headline and the answer tile match the approved renders (top region diff 0-5%, the difference being the new verdict controls); no card shows on a gist (heights shorter by design); empty-state rows 15-17 match in height at 1280 and +22 px at 390 (the new note under the controls), top region 0-2%. Row 18 (unseeded) as G11.

## Console
No errors from the change. `message-invalid` logs a browser warning that `abc` is not a valid number (pre-existing; the field is type=number). The failed states log `htmx:sendError` and `net::ERR_FAILED`, caused by the aborted request QA injects.

## Automated
`pytest tests/test_r084_gist.py test_r085_controls test_r087_list test_r090_feedback test_r091_focus test_r092_top_picks`: all pass.

Result: T-040, T-041, T-042, T-043, T-044, T-049, T-050 passes=true (already true in features.json, left unchanged).
