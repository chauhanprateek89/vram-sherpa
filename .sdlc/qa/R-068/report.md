# QA report: R-068 (round 1)

Result: **FAIL**. 6 of 7 criteria pass. Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027c`) in Chromium via Playwright at 390 and 1280 px, light and dark.

The QA agent could not write this file itself (its Write call was refused), so the orchestrator recorded its returned report here. Evidence is in this folder: `<state>-<w>-<theme>.png` (44 captures, kept local), `mockup-compare.txt`, `computed-styles.json`, `secondary-chip-link-probe-1280-light.json`, `control-audit.json`, `ua-default-baseline.json` and `favicon-and-logo-curl.txt`.

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Every R-067 screen is in "D layout + E glass", none in old styling | PASS | 44 captures match the mockup renders, 0.00% of pixels differing by more than 40. No legacy `.badge`, no overflow, no console errors |
| 2 | Shared elements have equal computed font, size, weight, colour, background, border and radius | **FAIL** | Secondary buttons: `Clear` (`.btn-clear`) is 15px/520, min-height 44; `Apply filters` (`.btn-secondary`, inside `<noscript>`) is 16px/560, min-height 48. Primary buttons, fields, same-class chips and the two disclosure summaries are equal |
| 3 | The classification label is identical on a card and on model detail | PASS | `.state-*`: 17px/600 pill, same colours in every width and theme |
| 4 | The gated marker is identical on a card and on model detail | PASS | `.tag-gated`: 14px/600, 99px radius, same fill. The wording `Gated` / `gated` is fixed by R-021 / R-036 |
| 5 | No control in the browser's default appearance | PASS | `control-audit.json`: every control has its own background, border, radius and font (Outfit) |
| 6 | The logo mark is in the header of every page | PASS | One `svg.mark` inside `a.brand` on every full page. The header strip differs 0.00% from the mockup |
| 7 | `<link rel="icon">` points to the approved favicon, served 200 from `/static/` | PASS | `favicon.svg?v=20261027c` returns 200 `image/svg+xml` and is byte-identical to `docs/design/marks/final-favicon.svg`. No route added |

## Judgement calls (not counted as failures)

These differ from criterion 2 read literally, but match the approved mockups:

- **Disclosure summaries:** the card's `How this is estimated` summary is 15px/520 with a 12px radius. The page-level Filters and Top Picks summaries are 16px/560.
- **Chips:** `.action-chip` (empty-state next steps) is 16px/560. Filter and example chips are 15px/520.
- **Links:** size follows their container (14, 15 and 17px). Colour, family, weight and underline are equal.

## Observation

Chrome's native number stepper shows on hover in number fields. The approved mockup doesn't hide it either.
