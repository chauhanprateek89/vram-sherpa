# Pipeline status

| | |
| --- | --- |
| Track | Full: UI/UX overhaul |
| Stage | Plan |
| Branch | sdlc/full-ui-overhaul |
| Gate waiting | none |
| Updated | 2026-10-25 |

## Your actions

- none

## Next step

Running in parallel: product applies the Gate 2 answers to R-067/R-069/R-071; architect revises ADR-0005 (styled GPU list chosen) and ADR-0006 (filters closed at both widths), then writes the plan into .sdlc/features.json. Then Build (sdlc:build workflow). **The release PR must be squash-merged** (Gate 2 repo-weight decision).

Follow-ups found at onboarding (not part of this track): host guard ignores APP_ENV/PUBLIC_HOSTNAME that docs/RELEASE_AND_DEPLOY.md tells the server to set (tunnel requests would get 400); release.yml pushes :latest before the smoke test; smoke test never hits /results; no Cache-Control on HTML (rule 9); local docker-compose.yml binds 8000 on all interfaces (rule 3).

## Log

| Date | Stage | Agents (model) | Outcome |
| --- | --- | --- | --- |
| 2026-10-25 | Bootstrap | orchestrator | Branch sdlc/onboard from main 96572b1; templates copied; old instruction file found: agent.md |
| 2026-10-25 | Map | sdlc:architect (inherit), sdlc:product (inherit) | HLD, project.json, init.sh (reached healthy), CLAUDE.md (34 lines), rules 11-21 proposed; 66 requirements (2 inferred); 7 agent.md contradictions; checks passed |
| 2026-10-25 | Gate 0 | user | Approved; R-052/R-065 confirmed; agent.md archived locally (gitignored); rules 11-21 accepted |
| 2026-10-25 | PR | orchestrator | Committed 8609bd9, pushed over HTTPS, opened PR #21; CI test passed |
| 2026-10-25 | Merge | orchestrator | User chose "Merge now"; PR #21 merged as f2ad2d8; release v0.1.51 succeeded (verify + release jobs); Onboard track done |
| 2026-10-25 | Intake | orchestrator | User goal is a UI/UX overhaul (Full track); scope answers recorded; tokens/sec dropped; branch sdlc/full-ui-overhaul from main f2ad2d8 |
| 2026-10-25 | Requirements | sdlc:product (inherit) | R-067 to R-079 planned (12 UI, 1 non-UI); 11 open questions for Gate 1 |
| 2026-10-25 | Gate 1 | user | Scope approved; Q-2 reword (not errors), Q-6 add example keep v0.1, Q-9 fix Cache-Control in this change; defaults accepted for the rest |
| 2026-10-25 | Requirements | sdlc:product (inherit) | Gate 1 answers folded in; R-080 added (Cache-Control); no open questions; 4 ambiguities logged in decisions |
| 2026-10-25 | Design | sdlc:architect (inherit) | HLD updated (proposed overhaul section, frozen interface); ADRs 0001-0007; open: htmx history-miss bug, rule 21 scope, worked-example figure |
| 2026-10-25 | Mockups r1 | sdlc:uiux (inherit) | 3 direction boards (A Basecamp, B Altimeter, C Expedition), 4 renders each + brand sheets; artifact published; all pass R-069/R-070 first-card targets |
| 2026-10-25 | Gate 2 r1 | user | All 3 directions rejected (wants modern bento/glass/spatial); htmx fix in scope; rule 21 covers all static/; worked example uses a real model |
| 2026-10-25 | Requirements | sdlc:product (inherit) | R-075 uses Llama 3.1 8B Instruct (7.33 GB); R-081 htmx history-miss added; open Q-12 (fallback if model leaves catalog), Q-13 (pushed partial URL breaks reload/share) |
| 2026-10-25 | Design | sdlc:architect (inherit) | Rule 21 amended (all static/); ADR-0007 revised (real model + fallback); ADR-0008 htmx history fix (HX-Push-Url to /results, refreshOnHistoryMiss, full page on history restore) also fixes reload/share; open: old partial links, dead controls after Back, Q-12 |
| 2026-10-25 | Mockups r1b | sdlc:uiux (inherit) | 3 new directions D Bento, E Glass, F Spatial with spec sheets and live states; artifact v2 at same URL; A-C archived in docs/design/rejected-r1/ |
| 2026-10-25 | Gate 2 r1 redo | user | Direction: D layout + E glass; redirect old partial links; controls-after-Back not selected; Q-12 keep unnamed |
| 2026-10-25 | Requirements | sdlc:product (inherit) | R-081 covers reload/share + 302/303/307 redirect of non-htmx /results/partials/content loads; R-032 corrected; R-075 fallback settled; R-068 records D+E hybrid; Q-12/Q-13 closed |
| 2026-10-25 | Design | sdlc:architect (inherit) | ADR-0008 revised: 302 redirect of non-htmx /results/partials/content loads, own Cache-Control/Vary, after host guard; HLD updated |
| 2026-10-25 | Mockups r2 | sdlc:uiux (inherit) | 24 screens/states x 4 renders (96 PNGs, 36 MB) in D+E hybrid from real app data; spec sheet, copy table (83 rows); merged logo; 0 contrast failures; artifact republished |
| 2026-10-25 | Gate 2 | user | Design approved; filters closed both widths + R-069 message-state allowance; styled GPU list; final-only PNGs, squash-merge |
