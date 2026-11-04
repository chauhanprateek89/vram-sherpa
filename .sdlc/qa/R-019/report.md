# R-019 Card ordering (T-036, T-037)
Scope: the two Won't fit ordering criteria (no UI change; no screenshots needed, mockups not applicable).

1. PASS. GET /results?vram_gb=24&context_tokens=4096&verdict=wont-fit&page=1..5 gave 24 cards. Overage is non-decreasing (4.4, 8.6, 20.9, 51.9 ... 1486.4). Positions 1-5, 16-20 and 21-24 match the names in the T-036 acceptance. At vram_gb=1 positions 6-10 are Qwen3-0.6B-Base, Llama-3.2-1B, Llama-3.2-1B-Instruct, gemma-3-1b-it, phi-1_5 (equal-overage ties by params desc, then name by code point). quant_bucket=Q8 starts with 20.9 GB over.
2. PASS. The overage is required - available of the open variant, unrounded. Five cards show "603.0 GB over" with true shortfalls 602.98, 602.98, 603.0, 603.0, 603.0, and they appear in that order, so the unrounded overage sets the order. The `_verdict_cards` key is the unrounded difference.
Tests: pytest tests/test_verdict_lists.py tests/test_results_view_routes.py: all pass.
