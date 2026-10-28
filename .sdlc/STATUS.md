# Pipeline status

| | |
| --- | --- |
| Track | Full (centred landing box) |
| Stage | Build (QA fix round 1) |
| Branch | sdlc/full-landing-box |
| Gate waiting | none |
| Updated | 2026-10-28 |

## Your actions

- Confirm the homelab runs v0.1.53 (digest `sha256:c1dacb73…`), e.g. `docker inspect` on the server; it can't be seen from outside because this release changes no app code.

## Next step

QA fix round 1 running: T-033 (opus) rebuilds the landing layout from the QA failure (`.sdlc/qa/R-083/report.md`). Then re-run QA for R-083 and the R-067/069/071/078 regression item only; the other three items passed. If R-083 still fails, ask the user (no second fix round). If this session ends first, `/sdlc:orchestrate resume`: check `T-033:` commits and `.sdlc/reviews/T-033.md`.
Then Release (`sdlc:release`, mode pr), Gate 3. Review nits to list in the PR: brittle source-text asserts in the T-025/T-027/T-030 tests, `_has_hardware_input` typed `object` (T-027), duplicated chip rules in styles.css (T-029), mktemp body file not removed in release.yml (T-032).

Other follow-ups still open (not part of this track): theme toggle and GPU input after an htmx cache restore, incl. a stale preset radio on a cache-hit Back; host guard ignores `APP_ENV`/`PUBLIC_HOSTNAME` that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); local compose binds 0.0.0.0:8000 (rule 3); `/docs`, `/redoc`, `/openapi.json` are public; font URL has no `?v` and the font is served as octet-stream.

## Log

Older rows: `.sdlc/log.md`.

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-28 | Intake | orchestrator | Full track: `/` becomes one centred GPU box (Claude new-chat style); box takes a VRAM number, context hidden on landing, no reverse lookup. Branch sdlc/full-landing-box from main 2aa63d0 |
| 2026-10-28 | Requirements | sdlc:product | R-083 (landing box) added; R-001, R-002, R-003, R-005, R-006, R-007, R-013, R-026, R-067, R-069, R-071, R-078 changed; R-082 done. Open Q-14 to Q-23. Committed 9aacb2e |
| 2026-10-28 | Gate 1 | user | Scope approved with defaults; Q-14 number rule on both pages; Q-17 submit + focus on load, no Clear on `/`; Q-23 add a /results smoke check in this change |
| 2026-10-28 | Requirements | sdlc:product (inherit) | R-062 amended for Q-23: smoke test also requires `GET /results?vram_gb=24&context_tokens=4096` → 200 with `id="summary-banner"` and `Manual VRAM (24 GB)`; Gate 1 answers column added to Q-14 to Q-23. Committed afaaf83 |
| 2026-10-28 | Design | sdlc:architect (inherit) | HLD section "Centred landing box", flows 9-10; ADRs 0009 (own landing.html), 0010 (302 /results→/), 0011 (gpu_input.py rule), 0012 (plain GET form + links, Clear is a link), 0013 (smoke /results check). 4 open points for Gate 2. Committed 34589c4 |
| 2026-10-28 | Mockups | sdlc:uiux (inherit) | L1-L4 as `final-landing-*` (6 mockups) + `final-results-from-landing`, 4 PNGs each; copy proposed; only `Clear` changes on /results; artifact https://claude.ai/artifact/S8yurugqYYCy6CNtbNTSbj. Committed 06ab377 |
| 2026-10-28 | Gate 2 | user | Design approved; proposed landing copy; L3 dashed inert GPU chip; focus on load at every width (kept despite Android keyboard) |
| 2026-10-28 | Plan | sdlc:architect (inherit) | T-023 to T-032 (opus: T-027 302 redirect, T-032 release smoke check; rest sonnet); UI tasks name their `final-landing-*` / `final-results-from-landing` mockups; static tasks bump ASSET_VERSION. No constitution conflicts, no cost. Committed 692c87f |
| 2026-10-28 | Build | sdlc:builder + sdlc:reviewer (sonnet; opus for T-027, T-032) | T-023 to T-032 all passed review (T-024, T-029 on the second review; no Opus rebuilds), commits c42116a..62ab9b1. Orchestrator check: lint, catalog, full pytest pass. Nits only (brittle source-text asserts, duplicated chip CSS, unremoved mktemp file) |
| 2026-10-28 | QA | sdlc:qa (inherit) ×5 | 4/5 pass (box rule + Clear, redirect/context/chips/unseeded, results regressions, R-062 smoke check). R-083 fails on layout: box 81 px off-centre at 1280, submit outside the box, grey filled field, no 16 px gutters at 390; 9-32% pixel diff vs approved PNGs. Behaviour, copy and focus pass. Sent back as T-033 (opus: unknown cascade cause, only fix round) |
