# R-087 QA report
All 13 criteria pass (curl on /results at the stated URLs; counts, headings, pager windows, aria-current, inert Previous/Next, page-size links).
Mockups: 24 full-page screenshots (6 states x 390/1280 x light/dark) in this folder; heights identical to docs/design/final renders, pixel diff 2-11% (background gradients/text AA), layout matches visually.
No horizontal scroll at 390, no tap target under 44 px in sizes/pager, 0 console errors. pytest tests/test_r087_list.py and test_results_view.py pass.
