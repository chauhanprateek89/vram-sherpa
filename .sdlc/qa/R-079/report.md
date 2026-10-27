# R-079 QA report (New styles reach returning visitors)

Date: 2026-10-27. App: http://localhost:8000 (running, current code). Branch sdlc/full-ui-overhaul. UI: no, no mockups.

## Criterion 1: WHEN the overhaul ships, request CSS and JS with an asset version other than `20261024a`

PASS.
- `src/vramsherpa/main.py:25`: `ASSET_VERSION = "20261027c"` (main branch has `20261024a`).
- `curl -fsS http://localhost:8000/` (saved as `index.html`), also /how-it-works, /results?..., /models/...: every page links
  `/static/css/styles.css?v=20261027c` and `/static/js/app.js?v=20261027c`. No page contains `20261024a`.
- Chrome DevTools network list for `/`: styles.css?v=20261027c [200], app.js?v=20261027c [200].
- Rule 21 satisfied: static/ changed (CSS, JS, favicons, font) and the version was bumped.
- Automated: new `tests/test_r079_asset_links.py` (10 tests, all pass) checks the version is not `20261024a`, CSS and JS links carry `?v=<ASSET_VERSION>` on four page types, and the old value is absent.

## Extra checks requested

1. Every `/static/` URL the pages link returns 200 (`static-urls.txt`):
   - /static/fonts/outfit-v1-latin-wght.woff2 200 (32292 bytes, content-type application/octet-stream)
   - /static/img/favicon.svg?v=20261027c 200; favicon-32.png?v=20261027c 200; apple-touch-icon.png?v=20261027c 200
   - /static/css/styles.css?v=20261027c 200; /static/js/app.js?v=20261027c 200
   - The same set is linked by /, /how-it-works, /results, /models/... (verified by grep of each page). Also covered by the new pytest.
2. Preload URL equals the font URL the stylesheet loads: yes. Preload href is `http://localhost:8000/static/fonts/outfit-v1-latin-wght.woff2` (no `?v`). styles.css `@font-face` has `url("../fonts/outfit-v1-latin-wght.woff2")`, which resolves to the same URL. `crossorigin` is present on the preload and the same-origin font request is CORS-anonymous, so they match. Only one font request is made per page load (Chrome DevTools reqid 2; Playwright request log).

## Font preload console warning: finding

Message seen: "The resource http://localhost:8000/static/fonts/outfit-v1-latin-wght.woff2 was preloaded using link preload but not used within a few seconds from the window's load event."
- Reproduced in the persistent Playwright browser profile on http://localhost:8000 for all 4 page types at 1440 and 390 px. Not shown on http://127.0.0.1:8000 in the same profile, and not shown in fresh isolated browser contexts (localhost or 127.0.0.1, two visits each), nor in the Chrome DevTools browser on first load of `/` (console empty). In every case `document.fonts` reported `Outfit:loaded` and only one woff2 request was made.
- After one `fetch('/static/css/styles.css?v=20261027c', {cache:'reload'})` in that profile, the warning stopped on all four page types. The text of the cached and fresh CSS was identical afterwards and both contain `@font-face`.
- Conclusion: not a defect in the markup (URL, `as`, `type`, `crossorigin` all correct). It comes from the persistent profile's HTTP cache for localhost:8000, which was populated by earlier QA runs against earlier builds. `/static` responses carry no `Cache-Control` (only ETag and Last-Modified), so browsers apply heuristic freshness and can reuse an old styles.css or font under an unchanged URL. The most likely trigger: the stylesheet entry in that cache did not use the preloaded font, so the preload went unused. That a single cache refresh cleared it is consistent with this, but I could not recover the stale entry to prove it exactly. A clean visitor does not see the warning.
- Observations for the orchestrator (not R-079 failures): (a) the font URL has no `?v=`, so a changed font file with an unchanged CSS URL would not be cache-busted; rule 21 only bumps the version on a change, which does change the CSS URL, but the font keeps its heuristic cache. (b) the woff2 is served as `application/octet-stream`, not `font/woff2`; browsers load it fine.

## Result

R-079: PASS. T-006 and T-008 marked `passes: true` in `.sdlc/features.json` (their other requirements, R-068 and R-077, are left to the orchestrator).

Evidence files: `.sdlc/qa/R-079/index.html`, `.sdlc/qa/R-079/static-urls.txt`. Test added: `tests/test_r079_asset_links.py`.
