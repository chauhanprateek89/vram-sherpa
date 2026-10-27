# Pipeline status

| | |
| --- | --- |
| Track | Quick: push `:latest` only after the smoke test passes |
| Stage | Plan |
| Branch | sdlc/quick-latest-after-smoke |
| Gate waiting | none |
| Updated | 2026-10-27 |

## Your actions

- none for this track. Still open from the last track: let the homelab pull v0.1.52 (watchtower does this once the server is set up).

## Next step

sdlc:architect (mode plan) adds a `track: quick` requirement for the release ordering in `docs/requirements.md` and writes the task(s) to `.sdlc/features.json` (CI config, so `opus`). Then Build (`sdlc:build`), QA, Release PR, Gate 3.

Other follow-ups still open (not part of this track): theme toggle and GPU input after an htmx cache restore, incl. a stale preset radio on a cache-hit Back; host guard ignores `APP_ENV`/`PUBLIC_HOSTNAME` that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); smoke test never hits /results; local compose binds 0.0.0.0:8000 (rule 3); `/docs`, `/redoc`, `/openapi.json` are public; font URL has no `?v` and the font is served as octet-stream.

## Log

Older rows: `.sdlc/log.md`.

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-27 | Release | sdlc:release (sonnet) | PR #22 opened, CI test passed |
| 2026-10-27 | Gate 3 | user | Merge now (squash); fix CLAUDE.md ASSET_VERSION line first |
| 2026-10-27 | Release | sdlc:release (sonnet) | PR #22 squash-merged as dc40346; release v0.1.52 passed (verify, build, smoke test incl. Cache-Control and static URLs, publish); live check not possible (no live URL); Full track done |
| 2026-10-27 | Intake | orchestrator | Quick track: push `:latest` only after the smoke test passes; branch sdlc/quick-latest-after-smoke from main dc40346; UI overhaul tasks archived to .sdlc/archive/sdlc-full-ui-overhaul.json; live URL recorded in project.json |
