# R-081 re-test for R-088 and R-084 (T-037, T-047)
- R-088 PASS: /results/partials/content?vram_gb=24&context_tokens=4096&family=qwen&family=llama&verdict=fits gives 302 to /results?...&verdict=fits (query byte for byte), ends 200 there; full page (1 header, 1 #search-form, 1 footer, stylesheet), 5 model cards; body byte-identical to direct /results (curl cmp) and card text/order/verdicts identical in browser. 1440 and 390 px. Screenshots r088-fits-redir-{1440,390}.png. No console errors.
- R-084 PASS: same address without verdict: 302 to /results?...family=llama, ends 200, full page, 0 cards, gist headline "fit or are tight", identical to direct. Screenshots r088-nov-redir-{1440,390}.png.
- Mockup comparison not repeated; earlier report.md covers layout.
