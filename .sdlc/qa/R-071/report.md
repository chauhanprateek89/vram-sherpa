# QA report: R-071 (input form on `/` and `/results`)

Result: **PASS** (19 of 19 criteria). Tested 2026-10-27 against http://localhost:8000 (assets `v=20261027d`) in Chromium via Playwright: 390×844 with touch and real taps, 1280×800, light and dark, JavaScript on and off.

The QA agent could not write this file itself (its Write call was refused), so the orchestrator recorded its returned report here. Evidence is in this folder: `fold.json`, `mockup-compare.txt`, `combobox-results.json` (101 checks), `messages-results.json` (112), `nojs-results.json` (48), `chips-presets.json`. Screenshots are kept local. Test added: `tests/test_r071_acceptance.py` (16 cases).

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Fields, presets and `Check compatibility` visible without scrolling on `/` | PASS | Lowest bottom edge at 598 of 844 px (390) and 495 of 800 px (1280). The same with JS off |
| 2 | Both input paths, chips and presets keep their behaviour | PASS | Chips and presets set and clear the right fields. Submit sends `gpu_id`, `gpu_search` and `context_tokens` |
| 3 | R-008 error messages word for word | PASS | All six messages appear. No message string in `main.py` changed since `f2ad2d8` |
| 4 | Headings and help text change only as the copy table says | PASS | The new hint and placeholder appear. The old strings and the in-form `How it works` link are absent |
| 5 | Mockup comparison (rule 10) | PASS | 32 renders. Page heights equal and 0.000% of pixels differ on 7 states. `gpu-suggest` differs 0.20–0.29%, only from the focus ring on the field |
| 6 | Error list shown once, in `role="alert"` | PASS | 5 error queries, one alert each, each message once |
| 7 | GPU feedback shown once, in `role="status"` | PASS | Ambiguous, not found and GB-hint queries. R-003 wording exact, nothing under the field |
| 8 | An in-place refresh shows only the current messages | PASS | Errors and GPU feedback clear after a fix, in all 4 width and theme runs |
| 9 | `#vram_gb` invalid state follows each refresh | PASS | Set on `abc`, cleared on a valid value, set again on a rejected value |
| 10 | The styled list replaces the native one, never both | PASS | `list` removed, combobox role set, `#gpu-listbox` inside `.combo` |
| 11 | Empty field lists every GPU in R-001 order and labels | PASS | 35 options, sorted by VRAM then name |
| 12 | Typing filters: case-insensitive, whitespace collapsed | PASS | `4060` and `  rtx   4060 ` both give the same 4 GPUs |
| 13 | Tap, click, or arrows plus Enter select | PASS | Each selects the exact label, sets `gpu_id`, empties VRAM and doesn't submit. Clear works |
| 14 | Escape closes the list and keeps the text | PASS | Tab closes it too |
| 15 | Options at least 44 px tall | PASS | 48–60 px at 390, 48 px at 1280 |
| 16 | `combobox`, `listbox` and `option` roles | PASS | Also `aria-expanded`, `aria-controls`, `aria-activedescendant` and `aria-selected` |
| 17 | Own JS only: no library, request or route added | PASS | 0 requests while typing and picking. No console errors |
| 18 | JS off: native list and free text work | PASS | Datalist holds 35 GPUs. Free text resolves on submit. Errors show and clear |
| 19 | Wording rules | PASS | See rows 3, 4, 6 and 7 |

## Notes (not failures)

1. **A rejected `0` in place:** the browser's own `min="1"` check stops a `0` from being sent. The rejected-value-in-place step was driven with `noValidate` on the form.
2. **When the list opens:** on typing or the Down arrow (ADR-0005), not when an empty field is tapped or focused.
3. **List over the button:** the open list overlays `Check compatibility`. Escape, Tab, a tap outside, or Enter with nothing highlighted gets past it.
4. **`24GB` chip:** it leaves the old GPU text in the field while clearing `gpu_id`. This is unchanged by this work.
5. **Test harness:** measurements now wait for `readyState=complete`. Waiting for `networkidle` alone sometimes measured the page before the htmx CDN script had run.
