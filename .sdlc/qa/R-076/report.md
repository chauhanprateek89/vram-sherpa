# QA report: R-076 (header and footer)

Result: **PASS** (5 of 5 criteria). Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027e`), at 390 px (touch) and 1280 px, light and dark, across all 23 page states. The QA agent's Write was refused, so the orchestrator recorded its returned report here. Test added: `tests/test_r076_acceptance.py` (22 tests).

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Header links, `aria-current` and the theme toggle (R-038/R-039) | PASS | Right on all 92 cells. Click, tap, Enter and Space all toggle. The choice persists in `vramsherpa-theme`, and the saved theme is applied before first paint |
| 2 | Logo mark and wordmark go to `/` | PASS | 34 px inline svg (`aria-hidden`) inside `a.brand[href="/"]`. Click and tap land on `/` |
| 3 | Footer values on every page (R-040) | PASS | Seeded: `2026-10-23`, `v0.1`, `2026-10-23`. Unseeded: `unseeded`, `v0.1`, `unseeded` |
| 4 | Phone footer readable | PASS | 3 stacked spans at 390 px, none clipped or overlapping, no sideways scroll |
| 5 | Header and footer identical on every screen | PASS | Same markup and box per width. 0.00% of pixels differ from the mockup on 92 renders (detail pages bottom-aligned) |

Console: 0 errors. One Chrome warning on the `vram_gb=abc` states, also present at `f2ad2d8`.

Notes, both unchanged from `f2ad2d8`:
- The toggle's `aria-label="Toggle color theme"` overrides its visible label.
- `app.js` stores the OS-derived theme on first load.
