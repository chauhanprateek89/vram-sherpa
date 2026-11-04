# R-024 QA (T-042)
Evidence: .sdlc/qa/R-024/<state>-<1280|390>-<light|dark>.png; compared with docs/design/final/gist-{reference,all-three,list-fits,message-invalid}-*.png (heights equal; remaining diff is the page background gradient only).
1. Ranking: PASS. pytest -k pick (23 tests) passes; order on the reference query matches (DeepSeek-Coder-V2-Lite-Instruct first).
2. Three cards: PASS. /results?vram_gb=24&context_tokens=4096 shows Best quality that fits, Best balance, Best small (DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-V2-Lite, Yi-1.5 9B Chat). Same three on verdict=fits pages 1 and 3.
3. Hidden: PASS. vram_gb=1 and min>max give no Top Picks.
4. Content: PASS. Name links to /models/<id>, "15.7B · Q4", description, plus verdict, "15.36 GB needed", "24.0 GB", "8.6 GB spare"; Best small shows Tight, 23.46 GB needed, 0.5 GB spare. Figures equal the card's fit pane (8.6 GB spare, 15.36 needed, 24.0). No gauge on picks.
Open state: gist `<details open>`; verdict selected `data-open-wide`.
Limit at 1280: first card verdict bottom 896 px (limit 900). Console: 0 errors.
