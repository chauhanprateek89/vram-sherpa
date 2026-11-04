# R-069 (R-084 change): layout limits on the live app

Scripts: measure_r084.py, order_check.py, pixdiff.py; data: measure_r084.json, mockup-compare-r084.txt; screenshots: gist/.
Measured in Chromium, light and dark identical. Bottom edges in CSS px from page top.

| Criterion | Result | Evidence |
|---|---|---|
| Top Picks after controls, before any list | PASS | reference: headline 117, controls 210, picks 358; with fits: list heading 691, card 741 |
| Reference 1280: headline, controls, 3 picks within 800 | PASS | headline 167, controls 286, picks 644 |
| Reference + fits 1280: first card name, verdict, margin within 900 | PASS | name 844, verdict 896, margin number 897 (mockup 896) |
| Message states 1280 within 900 | PASS | ambiguous 748, not found 703, input error 788 |
| Message states + fits 1280 within 1000 | FAIL | ambiguous verdict 974 (ok), not found 929 (ok), input error (`vram_gb=abc&verdict=fits`) verdict 1014, margin 1016 (limit 1000) |
| Phone, selected verdict, first card within 1688 | PASS | reference+fits 1471; ambiguous 1639; not found 1574; invalid 1634 |
| Phone gist, headline and controls within 1688 | PASS | headline 804; controls 949 (ref), 923, 1057, 1143, 1053, 1138 |
| First card bottom, reference+fits 1280 within 900 | PASS | fit-top (verdict) bottom 896, margin figure 897; whole card ends 1025 |

Other: no horizontal scroll (scrollWidth equals viewport everywhere); no console errors. Mockup pixel diff (top 1100 px): under 0.4 percent of pixels differ by more than 40 in light, up to 1.0 percent dark at 1280; 390 light about 1 percent, dark 2.3 to 3.9 percent (not a layout difference; mockups are viewport crops so heights differ).

The input-error fits state is 14 px over: the invalid gist is 1 px taller than the mockup (789 vs 788) and the mockup never drew this state with a card.

## Re-test after T-053 (1280, light and dark identical)
- invalid+fits: verdict top 994, margin 996 (limit 1000; was 1014/1016). Pass.
- Unchanged: invalid gist picks 788, ambiguous 748, not found 703, reference 644; reference+fits verdict 896, margin 897 (first card box bottom 1025); ambiguous+fits 964/966; notfound+fits 919/921.
- Phone 390: head 804, verdicts max 1143; +fits margin max 1630 (limit 1688). No console errors.
- Notices readable: gist/invalid-fits-notices-1280-dark.png, gist/invalid+fits-1280.png, gist/invalid-fits-notices-390-dark.png.
