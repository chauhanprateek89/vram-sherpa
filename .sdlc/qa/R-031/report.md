# R-031 QA
- PASS chips: /results?...family=qwen&quant_bucket=Q4&min_params_b=7&max_params_b=70&recommended_only=true shows "Active filters:" with Family: qwen, Quant: Q4, Min params >= 7B, Max params <= 70B, Recommended only (curl).
- PASS removal: each chip href drops only its value, keeps hardware/context/other filters, keeps verdict, drops page; page_size=10 kept (invalid 25 falls to default). Clicking Family chip in browser landed on /results?context_tokens=4096&vram_gb=24&quant_bucket=Q4. Chips are plain <a>, no hx-*.
- PASS Filters (2) with verdict=fits; pytest tests/test_r089_view_links.py passes.
- Console errors: 0. Screenshots: 1-1280.png, 1-390.png (no pixel-diff against mockup run; markup and text match).
