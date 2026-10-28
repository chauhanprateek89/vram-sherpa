# Pipeline status

| | |
| --- | --- |
| Track | Quick: push `:latest` only after the smoke test passes |
| Stage | Release PR |
| Branch | sdlc/quick-latest-after-smoke |
| Gate waiting | none |
| Updated | 2026-10-28 |

## Your actions

- none for this track. Still open from the last track: let the homelab pull v0.1.52 (watchtower does this once the server is set up).

## Next step

sdlc:release (mode pr) pushes the branch and opens the PR, then waits for CI. Then Gate 3 (repo merges squash). Open point for Gate 3: the version tag also waits for the smoke test (`docs/decisions.md`, 2026-10-27 Plan). After "Merge now": live check confirms the release run's step order (smoke test before "Tag the tested image as release and latest") and that `latest`, `v…` and `sha-…` share one digest, then flip T-022 `passes`.

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
