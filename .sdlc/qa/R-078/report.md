# QA report: R-078 (what stays)

Result: **PASS** (6 of 6 criteria). Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027e`), with `main` at `f2ad2d8` running from a temporary worktree (SQLite, same seeds) as the baseline, at 1440 and 390 px, JS on and off. The QA agent's Write was refused, so the orchestrator recorded its returned report here. Evidence: `behaviour-js.json` and `behaviour-nojs.json` in this folder; screenshots are kept local. Test added: `tests/test_r078_acceptance.py` (10 tests).

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Same models, order and figures as `main` for the two reference queries | PASS | The golden test passes. Live `extract_cards` output is identical on `main` and on the app: 71 cards / 284 panes, and 23 cards. 12 more queries also match |
| 2 | Same routes, URLs, controls and query parameters, except the R-081 redirect | PASS | Status codes match on 17 URL/header combinations. Every `hx-*` attribute is equal. The only addition is `inputmode="decimal"`. `main` pushed `/results/partials/content?…`; the app pushes `/results?…` (R-081) |
| 3 | Required, available and margin on every verdict (rule 16) | PASS | 343 quantisation clicks at 1280 and 390, 0 failures |
| 4 | Works without JavaScript | PASS | Submit, results, gauges, filters (`Apply filters`), free-text GPU and model-detail recompute all work. The ambiguous-GPU message shows once (`main` showed it twice) |
| 5 | Wording follows the copy table; protected text is unchanged | PASS | About 30 URLs and states diffed against `main`. Every Old string is replaced as the New column says. Errors, GPU messages, verdict labels, figures, labels, chips, link text, tags and footer are unchanged |
| 6 | Collapsed sections open without JS | PASS | Filters, Top Picks and each breakdown toggle natively |

Console: no errors or warnings.

## Wording changes the copy table does not list

These are not failures: all of them are in the approved Gate 2 mockups. They are listed in the release PR.

1. The theme button's text before JS runs, or with JS off, is `Dark mode` (it was `Theme`). With JS the labels are unchanged.
2. The card meta line has no visible `·` separator between params and family (the family is a tag now).
3. New `aria-label`s: `Main`, `Examples`, `Models`, `Variants`, and the description of the `/how-it-works` 10 GB example gauge.
4. `/how-it-works` drops the `:`/`,` punctuation between each term and its rule.
