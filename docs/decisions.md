# Decisions

## Decided by you

| Date | Gate | Decision |
| --- | --- | --- |
| 2026-10-25 | 0 | Approved the onboarding docs (`docs/requirements.md`, `docs/HLD.md`, `CLAUDE.md`, `.sdlc/`) as a true picture of the app. |
| 2026-10-25 | 0 | Confirmed both inferred requirements, R-052 (column change on an existing volume) and R-065 (first-time server setup and rollback); both are now `as-built`. |
| 2026-10-25 | 0 | Where `agent.md` and the code disagree (7 points), the code is right. `agent.md` is archived, not deleted. |
| 2026-10-25 | 0 | Accepted app-specific constitution rules 11 to 21 as binding. |
| 2026-10-25 | Intake | Run a UI/UX overhaul on the Full track. Screens: the whole app including the 390 px mobile layout, explicitly including /how-it-works. |
| 2026-10-25 | Intake | What drives it: the UI looks generic and dated, it is too dense (the answer gets lost), and it is bad on mobile. |
| 2026-10-25 | Intake | The designer proposes 2-3 distinct visual directions; the user picks one at Gate 2. |
| 2026-10-25 | Intake | Drop the tokens-per-second feature (branch `feat/tokens-per-second`, ee788b3). The branch is left in place, not deleted. |
| 2026-10-25 | 1 | Approved the overhaul scope, R-067 to R-079. |
| 2026-10-25 | 1 | Q-2 wording: headings, help text, notes and raw labels (such as `weights_gb`) may be reworded and repeats merged. Error messages, verdict labels and figures stay. New wording is approved in the Gate 2 mockups. |
| 2026-10-25 | 1 | Q-6 /how-it-works: restyle it and add plain-language terms plus a worked example (8B at Q4, 2048 tokens = 6.72 GB). Formulas are unchanged, so `ESTIMATION_POLICY_VERSION` stays v0.1. Reading of rule 16: the three change together only when the estimate itself changes. |
| 2026-10-25 | 1 | Q-9 rule 9: the overhaul ships with a `Cache-Control` header on HTML responses (a new UI: no requirement in this change). |
| 2026-10-25 | 1 | Accepted defaults: Q-1 the designer shows Top Picks placement options at Gate 2; Q-3 filters start collapsed on a phone with an active-filter count; Q-4 accept the proposed numbers (1688 px phone / 800 px desktop to the first card, 44 px taps, 12 px minimum text, 3:1 non-text contrast); Q-5 design and test at 390 and 1280 px; Q-7 each direction includes a logo mark and favicon; Q-8 restyle the controls only, but the native GPU suggestion list may be replaced if it works badly on phones; Q-10 fix only the two copy quirks (empty state "variants" vs models, Top Picks copy); Q-11 Gate 2 runs in two rounds: 2-3 direction boards (home/results at 390 and 1280, both themes), then every screen and state for the chosen direction. |

## Made during work

| Date | Stage | Decision | Why |
| --- | --- | --- | --- |
| 2026-10-25 | Bootstrap | Git remote operations use HTTPS through the gh credential helper (`.sdlc/project.json` `git`). | `origin` over SSH fails on the passphrase-protected key. |
| 2026-10-25 | Map | The architect added the missing `models.gated` column to the local dev database with `ALTER TABLE` instead of recreating the volume. | `init.sh` timed out on `UndefinedColumn`; the non-destructive fix from the README keeps local data. |
| 2026-10-25 | Gate 0 | `agent.md` was moved to `docs/archive/agent.md` but left gitignored, so it is archived locally and not committed. | It was deliberately gitignored and the repo is public; committing it would publish a file you kept out of git. Say so if you want it committed. |
