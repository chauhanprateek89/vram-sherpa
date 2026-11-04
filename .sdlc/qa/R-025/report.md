# R-025 QA (R-084 empty state)
- PASS: /results?vram_gb=1 and ?vram_gb=24&min_params_b=5&max_params_b=1 each contain "No models currently fit this setup." and "Try one of these next steps:", 0 cards (curl).
- PASS: reference query vram_gb=24 shows neither string.
- PASS: vram_gb=1&verdict=wont-fit shows the list (16 card matches), no empty state.
- PASS: mockup compare gist-empty-wont-link at 1280 and 390, light and dark: full-page heights equal (959, 1453); layout/content identical on visual check; differences only in background gradient and footer. Dark mockup diff ~ small after setting data-theme=dark. No horizontal scroll.
- PASS: console errors none; pytest tests/test_r084_gist.py passes.
Screenshots: .sdlc/qa/R-025/empty-{1280,390}-{light,dark}.png, wontfit-{1440,390}.png
