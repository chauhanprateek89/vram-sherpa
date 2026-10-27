# QA report: R-067

Result: **PASS** (3 of 3 criteria). Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027c`).

The QA agent could not write this file itself (its Write call was refused), so the orchestrator recorded its returned report here. Evidence is in this folder: 92 live captures `<name>-<390|1280>-<light|dark>.png`, `compare.txt`, `diffs/`, the `focus-states-*` contact sheets and `scripts/`.

## Method

- Chromium via Playwright at 390 and 1280 px, light and dark, against the 96 reference renders in `docs/design/final/`.
- Each live capture was compared with its render by pixel diff, and its visible text with the mockup HTML at the same width and theme.
- Extra `.model-card`s were hidden for the screenshots only, after measuring on the full page.
- Special setups:
  - **unseeded:** a scratch SQLite app (port 8101) with an empty database.
  - **detail-no-sources:** a scratch SQLite app (port 8102) with `models.sources` and `variants.sources` set to NULL.
  - **why-loading:** the `/results/why/**` request was held open.
  - **gpu-suggest:** the agent typed `4060` and pressed ArrowDown.

## Criteria

1. **Screens in "D layout + E glass", none in old styling: PASS.** Glass/bento on all 92 captures, and `data-theme` matched the requested theme. `scrollWidth` equals the viewport everywhere. No text is under 14 px and no tap target is under 44 px at 390. On the reference query: headline ends 167 / 804, first card name starts 624 / 1154, first verdict ends 770 / 1365 (1280 / 390).
2. **Every state matches its mockup at 390 and 1280, light and dark: PASS.** Every capture is the same size as its render. Pixel difference is 0.00% except the cases noted below.
3. **Row-by-row match for all 24 mockups: PASS.**

## Per state (390 light, 390 dark, 1280 light, 1280 dark; D = % of pixels differing)

| State | Result | Note |
| --- | --- | --- |
| home-default | PASS ×4 | D 0.00 |
| results-gpu | PASS ×4 | D 0.00 |
| results-manual-vram | PASS ×4 | D 0.00 |
| results-gpu-ambiguous | PASS ×4 | D 0.00 |
| results-gpu-notfound | PASS ×4 | D 0.00 |
| results-errors | PASS ×4 | D 0.00 |
| results-chips | PASS ×4 | D 0.00 |
| results-filters-open | PASS ×4 | D 0.02 to 0.06. Only the hover colour on `Filters (2)`, because the pointer stayed on it |
| results-picks-hidden | PASS ×4 | D 0.00 |
| results-tight-gated | PASS ×4 | D 0.00 |
| gpu-suggest | PASS ×4 | D 0.15 to 0.18. Only the focus ring on the field. No option is highlighted before ArrowDown, per ADR-0005 |
| why-closed | PASS ×4 | D 0.00 |
| why-loading | PASS ×4 | D 0.00 |
| why-loaded | PASS ×4 | D 0.00 |
| empty-links | PASS ×4 | D 0.00 |
| empty-inert | PASS ×4 | D 0.00 |
| empty-mixed | PASS ×4 | D 0.00 |
| unseeded | PASS ×4 | D 0.00 to 0.01. The agreed visible "i" on the dark info icon |
| detail-gated | PASS ×4 | D 0.00 |
| detail-ungated | PASS ×4 | D 0.00 |
| detail-no-sources | PASS ×4 | D 0.00 |
| detail-error | PASS ×4 | D 0.00 |
| how-it-works | PASS ×4 | D 0.00 |
| focus-states | PASS ×4 | Checked by keyboard on 5 live pages: 91 / 95 Tab stops (390 / 1280), each with a 3 px ring offset 2 px (1 px on the quantisation buttons), #4A17B8 light and #CDB9FF dark |

## Notes (not failures)

- **Console:** no errors. One warning on `results-errors` and `detail-error`: `The specified value "abc" cannot be parsed`, because the server echoes the rejected `vram_gb` into the number field. The pre-overhaul template does the same.
- **Hidden heading:** empty and unseeded pages carry a visually hidden `Results` heading that the mockup text lacks.
- **Agreed departures:** both show as agreed in `docs/decisions.md`: the dark unseeded "i" and the removed size-filter help lines.
