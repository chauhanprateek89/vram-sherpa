# Review: T-018 QA fix (R-070 round 1, criterion 3 flag)

## 2026-10-27

Commit reviewed: `cd5778973b2450aa510fbb16a7a9c6e7e9b75de2` "T-018: make model-page source links 44 px tap targets with no overlap (R-070 QA fix)" on `sdlc/full-ui-overhaul` (branch HEAD at review time).

Files changed: `src/vramsherpa/static/css/styles.css` (`.sources-list a`, line 552), `src/vramsherpa/main.py` (`ASSET_VERSION` `20261027d` to `20261027e`), `tests/test_ui_markup.py` (`test_source_links_are_44px_targets_that_never_overlap`, line 1514).

Verdict: **pass**

### Stage 1: spec compliance

R-070 (amended 2026-10-27): each link in the Sources list on `/models/<id>` is at least 44 CSS px tall at phone width, and neighbouring link boxes do not overlap. Decision rows: `docs/decisions.md` 2026-10-27 QA (R-070).

Method: real Chromium (playwright-core from the npx cache, VS Code server Node v24.18.1). Two scratch servers on one seeded SQLite DB (catalog 2026-10-23): HEAD on port 8121 (`styles.css?v=20261027e`) and the parent commit `cd57789^` on port 8122 (`?v=20261027d`), so before and after could be compared exactly. 390 px (mobile, touch) and 1280 px, light and dark, theme set through `localStorage` and `colorScheme`. Pages: Mistral NeMo (`model_mistral_nemo_12b_instruct_2407`, has the one-line URL `mistral.ai/news/mistral-nemo`), gated Llama 3.1 8B (`detail-gated` query) and Qwen 2.5 7B (`detail-ungated` query).

Sources list links (`.sources .sources-list a`):

| Page | Width | Before: min height, overlaps | After: min height, overlaps | scrollWidth |
| --- | --- | --- | --- | --- |
| Mistral NeMo (3 links) | 390 | 38.5 px, 2 pairs overlap by 10 px | 44 px (one-line), 65 px (two-line), 0 overlaps | 390 |
| Mistral NeMo | 1280 | 38.5 px, 2 pairs by 10 px | 44 px each, 0 overlaps | 1280 |
| Llama 3.1 8B, gated (2 links) | 390 | 61 px, 1 pair by 10 px | 65 px, 0 overlaps | 390 |
| Llama 3.1 8B, gated | 1280 | 38.5 px, 1 pair by 10 px | 44 px, 0 overlaps | 1280 |
| Qwen 2.5 7B (2 links) | 390 / 1280 | 61 / 38.5 px, 1 pair by 10 px | 65 / 44 px, 0 overlaps | 390 / 1280 |

The light and dark results are the same in every row. The criterion is met at phone width, and also at desktop width.

Per-variant provenance links (`.prov .sources-list a`, the same rule): 4 per page, each 58 to 62 px tall (the long llama.cpp URL wraps to two lines), 0 overlaps at both widths and in both themes. They look right. In side-by-side crops of the light 390 px and dark 1280 px renders, the link sits in the provenance block with 10 px more space above and below it. Nothing is clipped or misaligned.

Nothing else moved. I compared every landmark box (header, panes, h1, h2, table rows, footer) between HEAD and the parent. The model tile grows by exactly the Sources list's growth: +61.5 px (NeMo 390), +64.5 (NeMo 1280), +40 (gated or ungated 390), +43 (gated or ungated 1280). Everything below moves down by that amount, and no left position or width changes anywhere. The one other change is that each variant row is 20 px taller, because its provenance link lost the `-8px` margin (see nit 1). Against `docs/design/final/detail-{gated,ungated}-*`, the region above the Sources list is 0.00% different in all 8 renders. In light, the region between the Sources list and the table is 0.00% different once offset by the growth. Dark renders differ a little from the mockup below about 560 px even on the parent commit (parent vs mockup 1 to 3%, page background). That predates this commit and is not caused by it.

R-074's density limit still holds: on the ungated page at 390 px the first variant row starts at 1159 px (was 1119), well inside 1688.

Scope: only the source-link rule changed, plus the asset version and one test. The provenance links pick up the change because they share the selector. Nothing outside the task was added.

Test: `test_source_links_are_44px_targets_that_never_overlap` fails on the parent commit's stylesheet (checked by pointing the test module's `STYLES` at the parent's copy, repo untouched) and passes on HEAD. It checks `display: block`, `min-height: 44px`, that `.sources-list a` and `li` have no negative margin, and that `overflow-wrap: anywhere` is kept.

### Stage 2: code quality

No blocking issues.

- Constitution rule 21: the change is under `static/`, and `ASSET_VERSION` moved from `20261027d` to `20261027e`. The served URL carries `?v=20261027e`. Met.
- Constitution rule 10: this review looked at the change in a real browser against the mockups. R-070 still needs its QA round 2 before the task counts as done.
- The CSS is one rule plus a comment, in the file's one-line style. No dead code is left: the old negative-margin trick is gone completely.
- Security: CSS only. No markup, routes or data changed.

### Nits

1. `src/vramsherpa/static/css/styles.css:552`: the shared `.sources-list` selector also makes each per-variant provenance link 20 px taller, so each table row or phone sheet grows by 20 px (+80 px on a 4-variant page). It looks fine and is consistent, but the R-070 amendment and the decision row name only the Sources list. "Fixed" means either a line in `docs/decisions.md` noting that provenance links share the rule, or scoping the rule to `.sources .sources-list a`.
2. `src/vramsherpa/static/css/styles.css:552`: `display: block` makes the whole pane width clickable. At 1280 px a short URL's hit box is 660 px wide, so clicking blank space to the right opens an external tab. `width: fit-content` (or `inline-block` with `min-height: 44px` and no negative margin) would keep the box to the text and still meet the 44 px rule.
3. `docs/design/build/components.css:270` still has the old `inline-block; padding: 8px 0; margin: -8px 0`. This is an agreed departure, but a one-line note there or in `docs/design/README.md` would stop a mockup rebuild from bringing the old rule back.

### Test and lint results

- `pytest` (conda env `vram-sherpa`): 389 passed in 160.5 s.
- `ruff check .`: all checks passed. `ruff format --check .`: 26 files already formatted. `python tools/validate_catalog.py --data-dir data --config tools/catalog_sources.yaml`: passed (35 GPUs, 95 models, 380 variants).

Evidence (scratch, not in the repo): `/tmp/claude-1000/-home-pc-src-vram-sherpa/7cf73e3c-3e97-4db6-9f20-8a3c53563c0e/scratchpad/review-t018-qafix/` (`measure.mjs`, `measure.json`, full-page renders `<page>-<width>-<theme>-<head|parent>.png`, side-by-side crops `crop-nemo-*.png`).
