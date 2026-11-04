# R-032 QA report

- Changed (R-090) in-place refresh: PASS. Playwright at 1440: clicking Fits, page 2, size 10 each issued only `GET /results/partials/content?...` (no document load), a window marker survived, address became `/results?...` and cards updated (5, 5, 10). No console errors.
- Changed (R-088) parameters on /results and the three fragments: PASS. curl with `vram_gb=24&context_tokens=4096&verdict=fits&page=2&page_size=10`: /results 200 (10 cards), /results/partials/list 200 (10 cards), /results/partials/summary 200 (no cards), /results/partials/content 302 without HX-Request (R-081 redirect). Invalid `verdict=green&page=abc&page_size=7` gives 200 gist (0 cards). `page=99` and `page=15` show the same single card (last page).
- Also see R-088 and R-090 reports (mockup diffs, loading/failure states).
- Note: pytest full run did not finish within time limits in this session; not relied on.
