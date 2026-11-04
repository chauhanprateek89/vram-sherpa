# R-092 QA
Reference query /results?vram_gb=24&context_tokens=4096; screenshots ref-<1280|390>-<light|dark>.png.
1. PASS: picks show name, 15.7B · Q4, Fits/Tight, "15.36 GB needed", "24.0 GB", "8.6 GB spare" (Best small: Tight, 0.5 spare, 23.46 needed).
2. PASS: card pane for DeepSeek-Coder-V2-Lite-Instruct (verdict=fits page 1) shows Fits, 8.6 GB spare, 15.36 GB needed, 24.0 GB. pytest tests/test_r092_top_picks.py passes.
3. PASS: same three picks as before, no gauge/breakdown in picks; console errors: none.
4. PASS: verdict controls show only 71/0/24 counts, no figures.
5. PASS: pick heights equal the mockup: 1280: 233.56 x3; 390: 213.17, 194.28, 194.28. Gist ends 644 (mockup 644). Icon 18 px vs mockup 16 px as expected; no horizontal scroll.
