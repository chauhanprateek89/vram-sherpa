# QA report R-083 (T-028, T-029, T-030): FAIL

App at http://localhost:8000 (started with .sdlc/init.sh). `pytest tests/test_r083_*.py`: pass.
Screens: `.sdlc/qa/R-083/1-empty-*.png`, `2-suggest-*.png`, `3-pick-results-*.png`, `4-back-*.png`, and `shots/landing-<state>-<width>-<theme>.png` (24 renders, compared with `docs/design/final/`).

## Pass
- Copy word for word (h1, help line, placeholder); title `VRAM Sherpa`; one h1; box name `Search GPU`; submit name `Check compatibility`.
- Focus in box on load at 1280 and 390, both themes; one text input; no Clear, VRAM field or results text on `/`.
- Chips: 3 links, 44 px tall, in order `RTX 4060 8GB`, `8GB`, `24GB`; chip click gives /results?vram_gb=24&context_tokens=2048.
- Typing `4060` lists the four R-071 GPUs in order; click pick and ArrowDown+Enter go to /results (Hardware = NVIDIA GeForce RTX 4060 (8.0 GB), Context 2k).
- Blank and whitespace submit stay on `/`. `12 GB` gives `Manual VRAM (12 GB)`, Available VRAM 12; Clear is `a#clear-gpu` href `/`.
- `/?vram_gb=24&context_tokens=4096` answers 200 with no redirect.
- Back from /results (1280 and 390): one header, box, 3 chips, submit, one footer, styled; box empty.
- Unseeded (separate empty-DB instance on :8001): chips `Example GPU` (span, aria-disabled, dashed), `8GB`, `24GB`; status text present.
- No console errors.

## Fail
1. Box centring (Q-21, needs within 2 px): box centre is 81 px left of viewport centre at 1280 (box x=263, w=592) and 4.5 px off at 390.
2. Box width: 592 px at 1280 (mockup 754; the input is narrower than its `.combo`/form, which are 754, so the submit button sits outside the box instead of inset). At 390 the box is 381 px and the page has no side padding (pane and header flush near the edges; mockup has 16 px gutters and a 344 px box).
3. Mockup match: pixel diff (changed pixels, threshold 24) against docs/design/final: 1280 about 9 to 14 percent, 390 about 18 to 32 percent, in every state (empty, suggest, focus-box, focus-chip, focus-submit, unseeded) and both themes. Visible differences: box background is grey/filled not the dark inset field, h1-to-help-line gap is far larger, panel sits higher (1280: box centre y 478 vs mockup 428) and is taller, 390 layout shifted left with no gutters. See `1-empty-1280-dark.png` against `docs/design/final/landing-empty-1280-dark.png`, and `shots/landing-empty-390-light.png` against `docs/design/final/landing-empty-390-light.png`.
4. Chips and box fit without scrolling: yes (scrollHeight = viewport), so that bullet passes; vertical centre 478/800 (59.8%) and 458/844 (54.3%) are inside range.

Tasks T-028, T-029, T-030 left `passes: false`.
