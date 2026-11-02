# Pipeline status

| | |
| --- | --- |
| Track | Full (results gist first, then paginated by verdict) |
| Stage | Requirements |
| Branch | sdlc/full-results-gist |
| Gate waiting | none |
| Updated | 2026-11-02 |

## Your actions

- none

## Next step

Requirements running (sdlc:product) for the results-page change; then Gate 1. The previous track (landing box) finished with v0.1.54 live; its post-merge state files are now on this branch.

Other follow-ups still open (not part of this track): theme toggle and GPU input after an htmx cache restore, incl. a stale preset radio on a cache-hit Back; host guard ignores `APP_ENV`/`PUBLIC_HOSTNAME` that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); local compose binds 0.0.0.0:8000 (rule 3); `/docs`, `/redoc`, `/openapi.json` are public; font URL has no `?v` and the font is served as octet-stream; live 302 from `/results` arrives with an absolute https Location although the app sends a relative `/` (proxy rewrite, harmless); on `/results` the open suggestion list covers the submit until Escape; review nits from T-025/027/029/030/032 listed in PR #24.

## Log

Older rows: `.sdlc/log.md`.

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-28 | Requirements | sdlc:product | R-083 (landing box) added; R-001, R-002, R-003, R-005, R-006, R-007, R-013, R-026, R-067, R-069, R-071, R-078 changed; R-082 done. Open Q-14 to Q-23. Committed 9aacb2e |
| 2026-10-28 | Gate 1 | user | Scope approved with defaults; Q-14 number rule on both pages; Q-17 submit + focus on load, no Clear on `/`; Q-23 add a /results smoke check in this change |
| 2026-10-28 | Requirements | sdlc:product (inherit) | R-062 amended for Q-23: smoke test also requires `GET /results?vram_gb=24&context_tokens=4096` → 200 with `id="summary-banner"` and `Manual VRAM (24 GB)`; Gate 1 answers column added to Q-14 to Q-23. Committed afaaf83 |
| 2026-10-28 | Design | sdlc:architect (inherit) | HLD section "Centred landing box", flows 9-10; ADRs 0009 (own landing.html), 0010 (302 /results→/), 0011 (gpu_input.py rule), 0012 (plain GET form + links, Clear is a link), 0013 (smoke /results check). 4 open points for Gate 2. Committed 34589c4 |
| 2026-10-28 | Mockups | sdlc:uiux (inherit) | L1-L4 as `final-landing-*` (6 mockups) + `final-results-from-landing`, 4 PNGs each; copy proposed; only `Clear` changes on /results; artifact https://claude.ai/artifact/S8yurugqYYCy6CNtbNTSbj. Committed 06ab377 |
| 2026-10-28 | Gate 2 | user | Design approved; proposed landing copy; L3 dashed inert GPU chip; focus on load at every width (kept despite Android keyboard) |
| 2026-10-28 | Plan | sdlc:architect (inherit) | T-023 to T-032 (opus: T-027 302 redirect, T-032 release smoke check; rest sonnet); UI tasks name their `final-landing-*` / `final-results-from-landing` mockups; static tasks bump ASSET_VERSION. No constitution conflicts, no cost. Committed 692c87f |
| 2026-10-28 | Build | sdlc:builder + sdlc:reviewer (sonnet; opus for T-027, T-032) | T-023 to T-032 all passed review (T-024, T-029 on the second review; no Opus rebuilds), commits c42116a..62ab9b1. Orchestrator check: lint, catalog, full pytest pass. Nits only (brittle source-text asserts, duplicated chip CSS, unremoved mktemp file) |
| 2026-10-28 | QA | sdlc:qa (inherit) ×5 | 4/5 pass (box rule + Clear, redirect/context/chips/unseeded, results regressions, R-062 smoke check). R-083 fails on layout: box 81 px off-centre at 1280, submit outside the box, grey filled field, no 16 px gutters at 390; 9-32% pixel diff vs approved PNGs. Behaviour, copy and focus pass. Sent back as T-033 (opus: unknown cascade cause, only fix round) |
| 2026-10-28 | Build | sdlc:builder + sdlc:reviewer (opus) | T-033 passed first review, commit a80f5c6 (landing rules scoped under `.ask`, ASSET_VERSION 20261028f). Lint, catalog, full pytest pass. Nit: `.ask input[type=text]` repeats `.form-pane input[type=text]` (kept scoped on purpose) |
| 2026-10-28 | QA | sdlc:qa (inherit) ×2 | Re-run after T-033: R-083 passes (box 754 px centred, submit inset, 16 px gutters and 344 px box at 390; pixel diff now 1.2-1.6% at 1280, 2.8-3.9% at 390, suggest 5-9%); results regressions 0.00% diff. All 11 tasks pass |
| 2026-10-28 | Release | sdlc:release (sonnet) | PR #24 opened (pushed over HTTPS), CI `test` passed in 2m30s, head 632992f; `gh pr create` worked, no fallback needed |
| 2026-10-28 | Gate 3 | user | Merge now (squash); ship as is: no first-option highlight in the landing suggestion list, chips showing through the open list |
| 2026-10-28 | Merge | orchestrator | PR #24 squash-merged as 410e516; release run 40781009484 started |
| 2026-10-28 | Live check | sdlc:release (sonnet) | Release run 40781009484 success, v0.1.54; order held (build, smoke incl. new /results check, tag, publish); latest, v0.1.54, sha-410e516 share digest sha256:b358058f…; live `/` shows styles v=20261028f about 14 min after the run; landing box centred with focus at 1280 and 390, `12 GB` gives Manual VRAM (12 GB), empty /results 302 to /, Clear links to /, /healthz ok. Tasks archived; Full track done |
