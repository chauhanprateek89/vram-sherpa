# QA report: R-077 (accessibility), round 1

Result: **FAIL** on one item, criterion 6 (focus order at 390 px). The other criteria pass. Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027e`), at 390 px (touch) and 1280 px, light and dark. The QA agent's Write was refused, so the orchestrator recorded its returned report here. Test added: `tests/test_r077_acceptance.py` (15 tests).

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Text contrast 4.5:1 (3:1 large) in both themes, against the worst-case backdrop | PASS | 18,666 text boxes over 92 combinations, each with natural and worst-case backdrops: 0 failures. Lowest 6.29:1 light, 4.78:1 dark. The dark unseeded "i" is 15.6:1 |
| 2 | Non-text contrast 3:1 | PASS | Fields 3.50, buttons 4.30, gauge bar 3.87 against the pane and 4.20 against the track, reserve 6.49, markers 5.78 |
| 3 | Visible focus ring everywhere | PASS | 1,926 of 1,926 Tab stops show it: 3 px solid, offset 2 px (1 px on the quantisation buttons). Lowest 7.21:1 against neighbours. Matches the `focus-states` mockup |
| 4 | Reduced motion | PASS | 0 transitions or animations under `reduce`, against 1,205 or more normally. `getAnimations()` is 0 throughout |
| 5 | Reduced transparency and no-`backdrop-filter` fallbacks | PASS | Every pane is solid `--panel-solid` with no blur. 8,860 text boxes, 0 contrast failures |
| 6 | Keyboard operable, focus order matches the screen | **FAIL at 390 px** | Everything is operable by keyboard. But at 390 px the header DOM order is brand, `Estimator`, `How it works`, toggle, while the toggle is drawn on row 1 beside the brand, above the nav on row 2. Focus goes down to the nav, then back up to the toggle. The approved mockups have the same DOM order. 1280 px is fine |
| 7 | ARIA roles and states | PASS | 71 `.quants` groups with one pressed button each. 284 gauges with `role=img` and a label. One alert, one status and one `aria-current`. Combobox, listbox and option roles are present |

Observations (not criteria):
- The reserve outline over the bar measures 1.1–2.0:1, though it stays visible against the pane.
- The empty track against the pane is 1.02–1.48:1.
- Hover and active states were not tested.

---

# Round 2: re-evaluated against the amended R-077 (2026-10-27)

The user accepted the phone header focus order (`docs/decisions.md`, row "QA (R-077)"), and R-077's keyboard criterion now says so: at 390 px the header order is brand, `Estimator`, `How it works`, then the theme toggle. Round 1 recorded exactly that order (`keyboard-checks.txt`, `tab-reach.txt`). Everywhere else on the page, focus already followed the on-screen order. The code did not change, so no new run was needed.

**Criterion 6: PASS. R-077: PASS (7 of 7).**
