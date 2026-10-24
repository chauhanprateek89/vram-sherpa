# Decisions

## Decided by you

| Date | Gate | Decision |
| --- | --- | --- |
| 2026-10-25 | 0 | Approved the onboarding docs (`docs/requirements.md`, `docs/HLD.md`, `CLAUDE.md`, `.sdlc/`) as a true picture of the app. |
| 2026-10-25 | 0 | Confirmed both inferred requirements, R-052 (column change on an existing volume) and R-065 (first-time server setup and rollback); both are now `as-built`. |
| 2026-10-25 | 0 | Where `agent.md` and the code disagree (7 points), the code is right. `agent.md` is archived, not deleted. |
| 2026-10-25 | 0 | Accepted app-specific constitution rules 11 to 21 as binding. |

## Made during work

| Date | Stage | Decision | Why |
| --- | --- | --- | --- |
| 2026-10-25 | Bootstrap | Git remote operations use HTTPS through the gh credential helper (`.sdlc/project.json` `git`). | `origin` over SSH fails on the passphrase-protected key. |
| 2026-10-25 | Map | The architect added the missing `models.gated` column to the local dev database with `ALTER TABLE` instead of recreating the volume. | `init.sh` timed out on `UndefinedColumn`; the non-destructive fix from the README keeps local data. |
| 2026-10-25 | Gate 0 | `agent.md` was moved to `docs/archive/agent.md` but left gitignored, so it is archived locally and not committed. | It was deliberately gitignored and the repo is public; committing it would publish a file you kept out of git. Say so if you want it committed. |
