# QA report R-083 (T-028, T-029, T-030, T-033): PASS

Re-test after QA fix T-033. The app was rebuilt with `docker compose -f docker-compose.yml up -d --build`. GET `/` references `styles.css?v=20261028f`. Unseeded state was checked on a separate empty-SQLite instance of the same code on :8001. Renderer: Playwright headless Chromium. Screens: `.sdlc/qa/R-083/shots/landing-<state>-<width>-<theme>.png` (24), `pair-*.png` (live left, mockup right), `3-pick-results-1280-light.png`, metrics in `metrics.json`. The 1280 and 390 screens were each checked in light and dark.

## Previous failures: fixed
- 1280x800: box with its inset submit is 754 px wide at x=263, so its centre is exactly the viewport centre (offset 0). The submit (x=777, w=232) sits inside the box's right edge. Input font 20 px.
- 390x844: box is 344 px wide at x=23 (offset 0 from centre). The panes keep 16 px side gutters. Input font 18 px.
- Inset field look: the input is white in light and rgb(10,12,38) in dark, inside a bordered ring, as in the mockup. See `pair-empty-1280-light.png` and `pair-empty-390-light.png`.
- h1-to-help-line gap is 16 px (1280) and 14 px (390). Box vertical centre is 428 of 800 (53.5%) and 415.5 of 844 (49.2%). Both match the mockup, and both are inside the 35 to 65% range.

## Pixel diff against docs/design/final (changed pixels, threshold 24)
| Width | Previous run | This run |
| --- | --- | --- |
| 1280 | 9 to 14% | 1.15 to 1.63% (empty, focus-*, unseeded); suggest 5.04 to 5.10% |
| 390 | 18 to 32% | 2.77 to 3.94% (empty, focus-*, unseeded); suggest 8.76 to 8.84% |

The remaining non-suggest difference is text anti-aliasing and font hinting. The diff bands fall on glyph rows only (header, h1, help line, placeholder, chips, footer), and layout is identical in the pair images.

Visible differences, all in the suggest state (`pair-suggest-1280-light.png`):
1. The mockup draws option 1 highlighted (`aria-selected`, driven by its mock script). The live list highlights nothing until Down is pressed. This is behaviour, not layout, and is not a requirement failure.
2. In the live list the chips behind it show faintly through the list (more translucent than the mockup). Judged a small difference, not a missing element or layout change.

## Criteria (every bullet of R-083), at 1280 and 390, light and dark
- Pass: `/` returns 200. Header, box, 3 chips (`RTX 4060 8GB`, `8GB`, `24GB`), one submit and footer are present.
- Pass: no summary, results, Top Picks, filters, Available VRAM, context, assumed-8GB note or Clear on `/`. Exactly one text input.
- Pass: box name `Search GPU`, submit name `Check compatibility`, title `VRAM Sherpa`, one h1.
- Pass: copy word for word. h1 `Which local LLMs fit your GPU?`, help `Type a GPU name, or enter its VRAM, like 12 GB.`, placeholder `Your GPU, or its VRAM`. These match the approved mockup.
- Pass: no scrolling (scrollHeight minus viewport = 0 in all four combinations).
- Pass: centre offset 0 px at both widths. Vertical centre in range. Width 754 (at least 560) and 344 (at least 342). Text 20 and 18 px (at least 16).
- Pass: focus is in the box on load.
- Pass: `/?vram_gb=24&context_tokens=4096` gives 200 with no redirect (curl).
- Pass: typing `4060` shows 4 options. Clicking the first goes to `/results?gpu_search=NVIDIA+GeForce+RTX+4060+%288.0+GB%29`, and the page shows the RTX 4060 (8.0 GB) hardware. Keyboard pick passed in the previous run. Chip links carry `context_tokens=2048`.
- Pass: a blank box and a whitespace-only box stay on `/`. `12 GB` goes to `/results?gpu_search=12+GB` and shows `Manual VRAM (12 GB)`.
- Pass: Back shows one header, the box (empty), 3 chips and one footer.
- Pass: unseeded shows a dashed inert `Example GPU` span (aria-disabled), chips `8GB` and `24GB`, and the note `Catalog is not initialized yet.` Footer says `unseeded`.
- Pass: no console errors across all runs (error list empty).
- Pass: R-083 tests `pytest tests/test_r083_*.py` passed in the previous run. They were not re-run here.

T-028, T-029, T-030 and T-033 set to `passes: true`.
