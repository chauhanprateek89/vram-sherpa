# R-014 QA (T-036, T-039)
Checked with curl against http://localhost:8000, all pages walked at page_size=10.
1. No verdict, no card (R-084): PASS. /results?vram_gb=8&context_tokens=2048, ?vram_gb=24&context_tokens=4096 and ?vram_gb=1 render 0 `model-card` articles (only 3 Top Picks on the gist). A zero-count verdict (24 GB, verdict=tight) also shows no card.
2. Card only under a counted verdict (R-086): PASS. 8 GB/2048: fits 49, tight 9, wont-fit 37 cards, all distinct, no model in both a usable list and wont-fit. 24 GB: 71/0/24. 1 GB: 0/0/95. Card counts equal the control counts.
3. Headline `<N> model(s) fit or are tight`: PASS. 8 GB: 49+9 = "58 models"; 24 GB: "71 models"; 1 GB: "0 models"; ?vram_gb=3&family=phi: "1 model". Same text on the gist and on verdict views.
Tests: tests/test_verdict_lists.py, tests/test_r084_gist.py and headline tests in test_ui_markup.py pass. No screenshot/mockup diff taken; criteria are textual/count based.
