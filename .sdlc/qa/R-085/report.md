# R-085 QA (T-039, T-040, T-042)
All criteria PASS. Reference query /results?vram_gb=24&context_tokens=4096 at http://localhost:8000. Screenshots: .sdlc/qa/R-085/<gist|fits|wont|zero>-<1280|390>-<light|dark>.png
- Controls in order Fits, Tight, Won't fit, count + label, one set only: PASS (71 / 0 / 24; group "Show models by verdict").
- Colour tokens and icon shapes (#i-fits, #i-tight, #i-wont), green/amber/red tiles: PASS.
- Select a verdict >0 shows list on page 1: PASS (Fits: 5 cards, "showing 1-5 of 71").
- One verdict at a time; switching replaces list on page 1 keeping page_size (verdict=fits&page_size=10&page=2 -> Won't fit gives verdict=wont-fit&page_size=10, 1-10 of 24): PASS.
- Selecting the selected control returns to gist, URL /results?vram_gb=24&context_tokens=4096 with no verdict/page/page_size: PASS.
- Selected marked by more than colour (aria-current=true, 3px border vs 1px): PASS.
- Counts stay 71/0/24 across verdict, page, size: PASS.
- Tight at 0 is span.vt, dashed, tabindex -1, no href, text "0 Tight", absent from tab order: PASS.
- verdict=tight at 0: HTTP 200, gist, role=status "No models in Tight with these settings.": PASS.
- Top Picks: closed at 899 and 390, open at 1280, data-open-wide present, same three picks as gist: PASS.
- Note "Pick a quantisation on any card to see how it changes." directly under the controls with a verdict selected: PASS.
- Mockups (list-fits, verdict-zero, 1280/390, light/dark): page sizes identical; pixel diff 0.1-4.7% (>40 level), differences are the body background gradient and anti-aliasing, layout/content match visually (fits 1280 light checked). First card top at 741 px, within the 900 limit.
- Console errors: none. pytest tests/test_r085_controls.py: 9 passed.
Note: vertical-bar slug is verdict=wont-fit; verdict=wont_fit is ignored and shows the gist (invalid value).
