# QA R-078 (changed bullets R-084, R-090), 2026-11-04

Result: PASS (2 of 2). Tasks T-038, T-039, T-046 marked passes true.

1. Reference queries walked at page_size=10, verdict=fits then tight, every page: script `walk_r084.py`, output `walk_r084.out`. Both queries identical to the f2ad2d8 golden fixture (`tests/fixtures/r078_reference.json`): 71 models / 284 panes and 23 models, same order, verdicts, required, available, margin. Gist has 0 cards. pytest (test_reference_queries, r078, r089_apply_view, r084_gist): all pass, 2 skipped.
2. JavaScript disabled (Playwright context javaScriptEnabled=false, 1280 and 390): landing form Enter goes to /results?gpu_search=24 (200), gist shows "71 models fit or are tight", 0 cards; verdict link shows 5 cards / 20 gauges; page=2 link 5 cards / 20 gauges; page_size=10 link 10 cards; `Apply filters` (name=apply_view value=fits:10) redirects to ...&verdict=fits&page_size=10 with 10 cards. Screenshots: nojs-gist/fits/applied-{1280,390}.png.

Not done: pixel-diff of mockups (criteria are behavioural).
