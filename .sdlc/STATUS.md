# Pipeline status

| | |
| --- | --- |
| Track | Full (centred landing box) |
| Stage | Plan |
| Branch | sdlc/full-landing-box |
| Gate waiting | none |
| Updated | 2026-10-28 |

## Your actions

- Confirm the homelab runs v0.1.53 (digest `sha256:c1dacb73…`), e.g. `docker inspect` on the server; it can't be seen from outside because this release changes no app code.

## Next step

Gate 2 approved. Next session (`/sdlc:orchestrate resume`):
1. Plan: `sdlc:architect`, mode plan. Tasks from R-083 and the changed requirements (incl. R-062 smoke check, ADRs 0009-0013) into `.sdlc/features.json`, a model on each; UI tasks name their mockup (`docs/design/mockups/final-landing-*.html`, `final-results-from-landing.html`). Approved copy and choices: `docs/decisions.md` Gate 2 rows (2026-10-28).
2. Build (`sdlc:build` workflow), QA (`sdlc:qa`, compare with the approved mockups), Release, Gate 3.

Other follow-ups still open (not part of this track): theme toggle and GPU input after an htmx cache restore, incl. a stale preset radio on a cache-hit Back; host guard ignores `APP_ENV`/`PUBLIC_HOSTNAME` that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); smoke test never hits /results; local compose binds 0.0.0.0:8000 (rule 3); `/docs`, `/redoc`, `/openapi.json` are public; font URL has no `?v` and the font is served as octet-stream.

## Log

Older rows: `.sdlc/log.md`.

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-27 | Intake | orchestrator | Quick track: push `:latest` only after the smoke test passes; branch sdlc/quick-latest-after-smoke from main dc40346; UI overhaul tasks archived to .sdlc/archive/sdlc-full-ui-overhaul.json; live URL recorded in project.json |
| 2026-10-27 | Plan | sdlc:architect (inherit) | R-082 added (track: quick): `latest` and the version tag move only after the smoke test; `sha-` pushed before; tag the tested digest with `imagetools create`. T-022 (opus). No UI, architecture or cost flags. Committed 4de1e2b |
| 2026-10-28 | Build | sdlc:builder (opus), sdlc:reviewer (opus) | T-022 passed: commit b7452ed, passed first review; nits: test's version-tag regex is line-bound (build tags checked exactly, so not blocking), watchtower arrow off-centre in the deploy doc diagram |
| 2026-10-28 | QA | orchestrator | Browser QA skipped (R-082 has nothing visible in the app). Local checks: YAML parses, lint, catalog, full pytest pass; new test 6/6 on the fix, 4/6 fail on the old workflow. Real proof is the first release run |
| 2026-10-28 | Release | sdlc:release (sonnet) | PR #23 opened (pushed over HTTPS), CI `test` passed; `gh pr edit` fails on the Projects (classic) GraphQL deprecation, body updated via `gh api` |
| 2026-10-28 | Gate 3 | user | Merge now (squash); version tag waits for the smoke test too (kept as built) |
| 2026-10-28 | Merge | orchestrator | PR #23 squash-merged as 2aa63d0; release run 40666348177 started |
| 2026-10-28 | Live check | sdlc:release (sonnet) | Release run 40666348177 success, v0.1.53; step order verified (build, smoke, tag, publish); only sha-2aa63d0 pushed by build; latest, v0.1.53, sha-2aa63d0 share digest sha256:c1dacb73…; live site / 200, /healthz ok (catalog 2026-10-23), styles v=20261027e. Homelab image not checkable from outside. T-022 passes; tasks archived; Quick track done |
| 2026-10-28 | Intake | orchestrator | Full track: `/` becomes one centred GPU box (Claude new-chat style); box takes a VRAM number, context hidden on landing, no reverse lookup. Branch sdlc/full-landing-box from main 2aa63d0 |
| 2026-10-28 | Requirements | sdlc:product | R-083 (landing box) added; R-001, R-002, R-003, R-005, R-006, R-007, R-013, R-026, R-067, R-069, R-071, R-078 changed; R-082 done. Open Q-14 to Q-23. Committed 9aacb2e |
| 2026-10-28 | Gate 1 | user | Scope approved with defaults; Q-14 number rule on both pages; Q-17 submit + focus on load, no Clear on `/`; Q-23 add a /results smoke check in this change |
| 2026-10-28 | Requirements | sdlc:product (inherit) | R-062 amended for Q-23: smoke test also requires `GET /results?vram_gb=24&context_tokens=4096` → 200 with `id="summary-banner"` and `Manual VRAM (24 GB)`; Gate 1 answers column added to Q-14 to Q-23. Committed afaaf83 |
| 2026-10-28 | Design | sdlc:architect (inherit) | HLD section "Centred landing box", flows 9-10; ADRs 0009 (own landing.html), 0010 (302 /results→/), 0011 (gpu_input.py rule), 0012 (plain GET form + links, Clear is a link), 0013 (smoke /results check). 4 open points for Gate 2. Committed 34589c4 |
| 2026-10-28 | Mockups | sdlc:uiux (inherit) | L1-L4 as `final-landing-*` (6 mockups) + `final-results-from-landing`, 4 PNGs each; copy proposed; only `Clear` changes on /results; artifact https://claude.ai/artifact/S8yurugqYYCy6CNtbNTSbj. Committed 06ab377 |
| 2026-10-28 | Gate 2 | user | Design approved; proposed landing copy; L3 dashed inert GPU chip; focus on load at every width (kept despite Android keyboard) |
