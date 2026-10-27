# Review: T-012 QA fix (R-068 round 1, criterion 2)

## 2026-10-27

Commit reviewed: `1ae19b26f6eff1c974e743c9555e53d0bdfc8246` "T-012: make Apply filters match Clear (R-068 QA fix)" on `sdlc/full-ui-overhaul`. Branch HEAD at review time: `3f78376` (later commits only add QA evidence and a decisions row).

Files changed: `src/vramsherpa/static/css/styles.css` (`.btn-secondary`, line 332), `src/vramsherpa/main.py` (`ASSET_VERSION`), `tests/test_ui_markup.py` (`_declared`, `test_secondary_buttons_share_size_weight_and_height`).

Verdict: **pass**

### Stage 1: spec compliance

R-068 criterion 2 (shared elements compute the same font size, weight, colour, background, border and radius) for the secondary buttons `Clear` (`.btn-clear`) and `Apply filters` (`.btn-secondary`, inside `<noscript>`):

Checked in real Chromium (Playwright 1.x, VS Code server Node v24.18.1) against the running app at `http://localhost:8000/results?vram_gb=24&context_tokens=4096`, with JavaScript off, at 390 and 1280 px, light and dark. The served stylesheet is `styles.css?v=20261027d` and is byte-identical to the working tree. Values were read with CDP `CSS.getComputedStyleForNode`.

| Property | Clear | Apply filters | Equal in all 4 runs |
| --- | --- | --- | --- |
| font-family | Outfit stack | Outfit stack | yes |
| font-size | 15px | 15px | yes |
| font-weight | 520 | 520 | yes |
| min-height | 44px | 44px | yes |
| color | light `rgb(20,17,43)` / dark `rgb(243,241,255)` | same | yes |
| background | light `rgba(255,255,255,.6)` / dark `rgba(0,0,0,.3)`, no image | same | yes |
| border | solid, same width, light `rgb(100,95,150)` / dark `rgb(142,139,196)` | same | yes |
| radius | 16px | 16px | yes |

The only differences are `padding-left` (16 vs 20 px) and `min-width` (76 vs 0 px), neither of which is in R-068's list. Rendered height is 52 px for Clear and 44 px for Apply filters: Clear stretches to the 52 px search field in `.field-row` (flex, default `align-items: stretch`), as in the approved mockups. Computed `min-height` is equal.

- Test would fail on drift: confirmed by running the new test against altered copies of the stylesheet (repo untouched). It fails on the pre-fix CSS, on Clear's font-size changed in the shared rule, on Apply's weight back to 560, on Apply's min-height back to 48px, and on Apply's font-size removed. See the gaps under nits.
- `ASSET_VERSION`: `20261027c` to `20261027d`, a new value not used before in history (rule 21). The running app serves `?v=20261027d`.
- Scope: one CSS rule, the version bump and one test. Nothing beyond the task.
- Mockups: `.btn-clear` is unchanged, so no approved screen that shows Clear changes. See the follow-up below about `focus-states`.

Spec issues: none blocking.

Follow-up for the orchestrator (not a code defect): `docs/decisions.md` line 53 says `Apply filters` "appears in no mockup". It does appear in the approved Gate 2 state 24, `docs/design/mockups/final-focus-states.html` (around line 416, `<button class="btn-secondary is-focus">Apply filters</button>`), drawn at the old 16px/560/48px. That state is judged on focus rings (R-067 QA checked rings by keyboard), and those are unaffected. But the button now differs in size from that render. Fixed means correcting the decision row to name this deliberate departure from state 24 and listing it for Gate 3, as the T-020 row does. No code change is needed: making Clear match Apply instead would change many approved screens.

### Stage 2: code quality

- Correctness: the rule now reuses the existing tokens `var(--tap)` and `var(--fs-small)`, the same as `.btn-clear`. Hover (`var(--inset)`) and transition already matched.
- Constitution: rule 10, the browser check above (QA should still re-run R-068 criterion 2). Rule 21, the bump is done. Rule 18, no CI check is touched. No security surface: CSS and a test only.
- Fit: the test helper `_declared` sits next to `_rules` and reads like the surrounding tests.

Blocking quality issues: none.

### Nits

1. `tests/test_ui_markup.py` (`test_secondary_buttons_share_size_weight_and_height`) checks only font-size, font-weight and min-height. Changing `.btn-secondary` to `border-radius: var(--r-pill)` or `color: var(--muted)` still passes the test, yet R-068 lists colour, background, border and radius too. Fixed: also compare `color`, `background`, `border` and `border-radius`. Radius must be read from the `.btn-clear` override rule, which sets `var(--r-in)` after the shared rule's `var(--r-pill)`.
2. Same test: `_declared` reads only the rule whose selector is exactly `.btn-clear, .chips .chip`. Adding `font-size: 14px` to the separate `.btn-clear { ... }` rule (line 324) makes Clear 14px, but the test still passes. Fixed: merge the values from both `.btn-clear` rules, last one wins.
3. `tests/test_ui_markup.py` `_declared`: the token lookup takes the first `--x:` definition in the file. That is fine today because `--tap` and `--fs-small` are defined once, but it would misread a token that a theme block redefines. Optional: assert that the token is defined only once.
4. `styles.css:326` vs `:334`: the press scale differs (`.btn-clear:active` 0.95, `.btn-secondary:active` 0.97). This is not in R-068's list. Align it only if the designer wants that.

### Test and lint results

- `pytest` (conda env `vram-sherpa`): 359 passed in 262.82s.
- `ruff check .`: all checks passed. `ruff format --check .`: 24 files already formatted. `python tools/validate_catalog.py --data-dir data --config tools/catalog_sources.yaml`: passed (35 GPUs, 95 models, 380 variants).
- Browser probe and mutation scripts, outputs and screenshots are kept outside the repo, in the reviewer scratchpad (`t012-qa-fix/`).
