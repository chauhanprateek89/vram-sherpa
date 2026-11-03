# Pipeline status

| | |
| --- | --- |
| Track | Full (results gist first, then paginated by verdict) |
| Stage | Gate 2 |
| Branch | sdlc/full-results-gist |
| Gate waiting | Gate 2: design and mockups for R-084 to R-092 |
| Updated | 2026-11-03 |

## Your actions

- Answer Gate 2 (design and mockups): https://claude.ai/artifact/6WMqjKd79PuxAuKagxBWZA

## Next step

Gate 2 is waiting. Design e56d9b6 (ADRs 0014 to 0019), mockups c289fca, gallery https://claude.ai/artifact/6WMqjKd79PuxAuKagxBWZA. Open points and build notes are in `docs/design/README.md`. After approval: Plan (sdlc:architect, mode plan).

Other follow-ups still open (not part of this track): theme toggle and GPU input after an htmx cache restore, incl. a stale preset radio on a cache-hit Back; host guard ignores `APP_ENV`/`PUBLIC_HOSTNAME` that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); local compose binds 0.0.0.0:8000 (rule 3); `/docs`, `/redoc`, `/openapi.json` are public; font URL has no `?v` and the font is served as octet-stream; live 302 from `/results` arrives with an absolute https Location although the app sends a relative `/` (proxy rewrite, harmless); on `/results` the open suggestion list covers the submit until Escape; review nits from T-025/027/029/030/032 listed in PR #24; R-083 and the requirements it changed still read `planned` in docs/requirements.md although v0.1.54 shipped them.

## Log

Older rows: `.sdlc/log.md`.

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-28 | Build | sdlc:builder + sdlc:reviewer (opus) | T-033 passed first review, commit a80f5c6 (landing rules scoped under `.ask`, ASSET_VERSION 20261028f). Lint, catalog, full pytest pass. Nit: `.ask input[type=text]` repeats `.form-pane input[type=text]` (kept scoped on purpose) |
| 2026-10-28 | QA | sdlc:qa (inherit) ×2 | Re-run after T-033: R-083 passes (box 754 px centred, submit inset, 16 px gutters and 344 px box at 390; pixel diff now 1.2-1.6% at 1280, 2.8-3.9% at 390, suggest 5-9%); results regressions 0.00% diff. All 11 tasks pass |
| 2026-10-28 | Release | sdlc:release (sonnet) | PR #24 opened (pushed over HTTPS), CI `test` passed in 2m30s, head 632992f; `gh pr create` worked, no fallback needed |
| 2026-10-28 | Gate 3 | user | Merge now (squash); ship as is: no first-option highlight in the landing suggestion list, chips showing through the open list |
| 2026-10-28 | Merge | orchestrator | PR #24 squash-merged as 410e516; release run 40781009484 started |
| 2026-10-28 | Live check | sdlc:release (sonnet) | Release run 40781009484 success, v0.1.54; order held (build, smoke incl. new /results check, tag, publish); latest, v0.1.54, sha-410e516 share digest sha256:b358058f…; live `/` shows styles v=20261028f about 14 min after the run; landing box centred with focus at 1280 and 390, `12 GB` gives Manual VRAM (12 GB), empty /results 302 to /, Clear links to /, /healthz ok. Tasks archived; Full track done |
| 2026-11-02 | Requirements | sdlc:product (inherit) | R-084 to R-092 added (gist first, verdict controls, verdict lists, pages and size, address, filters, in-place loading, accessibility, Top Picks figures); R-013, 014, 016, 019, 022, 023, 024, 025, 031, 032, 067, 069, 078, 080, 081 changed. Open Q-24 to Q-42; rules 16 and 18 need rulings (Q-39, Q-42). Committed 93eca4f |
| 2026-11-02 | Gate 1 | user | Scope approved; Q-28 sizes 5 and 10, default 5; Q-29 Previous, numbers, Next; Q-31 changed to closest to fitting first; Q-39 and Q-42 yes; other proposals stand |
| 2026-11-02 | Requirements | sdlc:product (inherit) | Gate 1 answers applied: R-019, R-086 `Won't fit` closest to fitting first (overage, then parameter count descending, then name); R-062 gains the `verdict=fits` smoke check; Q-24 to Q-42 answered. Committed 920bbe0 |
| 2026-11-02 | Design | sdlc:architect (inherit) | HLD section "Results gist and verdict lists", flows 11-14, `results_view.py`; ADRs 0014 (view in query string), 0015 (one card list, sorted on the server), 0016 (form keeps view, `apply_view` 302 without JS), 0017 (htmx history cache off, amends 0008), 0018 (links refreshed in place, feedback outside swap), 0019 (`verdict=fits` smoke check). No formula change, no cost, no constitution conflict. 2 open points for Gate 2. Committed e56d9b6 |
| 2026-11-03 | Mockups | sdlc:uiux (sonnet) | G1 to G13 as `final-gist-*` (26 mockups, 104 PNGs), no copy proposed, gallery published. Stopped once by the usage limit and resumed. Scroll width, contrast and 44 px targets pass; `verdict=fits` first card ends at 896 px against the 900 limit. Committed c289fca |
