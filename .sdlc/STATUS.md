# Pipeline status

| | |
| --- | --- |
| Track | Full: UI/UX overhaul |
| Stage | Gate 2 round 1 (redo) |
| Branch | sdlc/full-ui-overhaul |
| Gate waiting | Gate 2 round 1: pick a direction (D Bento, E Glass, F Spatial) at https://claude.ai/artifact/GpFFFNrx4duQTg63MYdQQQ; htmx follow-ons; Q-12 |
| Updated | 2026-10-25 |

## Your actions

- Pick a direction at https://claude.ai/artifact/GpFFFNrx4duQTg63MYdQQQ and answer the open points.

## Next step

Wait for the pick. Then uiux round 2 (every R-067 screen and state in the chosen direction; fix mockup counts so the Tight count matches the cards per R-022) in parallel with product folding the htmx/Q-12 answers into R-075/R-081. Then Gate 2 proper (incl. Q-2 copy approval).

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
