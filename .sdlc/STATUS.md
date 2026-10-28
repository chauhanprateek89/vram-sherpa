# Pipeline status

| | |
| --- | --- |
| Track | Full (centred landing box) |
| Stage | Gate 1 |
| Branch | sdlc/full-landing-box |
| Gate waiting | Gate 1 (scope: R-083 plus changes to R-001, R-002, R-003, R-005, R-006, R-007, R-013, R-026, R-067, R-069, R-071, R-078; open Q-14 to Q-23) |
| Updated | 2026-10-28 |

## Your actions

- Confirm the homelab runs v0.1.53 (digest `sha256:c1dacb73…`), e.g. `docker inspect` on the server; it can't be seen from outside because this release changes no app code.

## Next step

Gate 1: the user approves the scope and answers Q-14 to Q-23 (defaults written into the requirements). Then Design (`sdlc:architect`, mode design) and Mockups (`sdlc:uiux`, landing states L1 to L4).

Other follow-ups still open (not part of this track): theme toggle and GPU input after an htmx cache restore, incl. a stale preset radio on a cache-hit Back; host guard ignores `APP_ENV`/`PUBLIC_HOSTNAME` that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); smoke test never hits /results; local compose binds 0.0.0.0:8000 (rule 3); `/docs`, `/redoc`, `/openapi.json` are public; font URL has no `?v` and the font is served as octet-stream.

## Log

Older rows: `.sdlc/log.md`.

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-27 | Release | sdlc:release (sonnet) | PR #22 opened, CI test passed |
| 2026-10-27 | Gate 3 | user | Merge now (squash); fix CLAUDE.md ASSET_VERSION line first |
| 2026-10-27 | Release | sdlc:release (sonnet) | PR #22 squash-merged as dc40346; release v0.1.52 passed (verify, build, smoke test incl. Cache-Control and static URLs, publish); live check not possible (no live URL); Full track done |
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
