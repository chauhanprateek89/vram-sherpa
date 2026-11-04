# R-086 QA report

App at http://localhost:8000 (200). Full `pytest` run: all passed. `tests/test_verdict_lists.py` already encodes the worked examples.

| Criterion | Result | Evidence |
|---|---|---|
| Fits/Won't fit totals equal the control counts (71 / 24) | PASS | Controls show 71 / 0 / 24; `verdict=fits` pages 1-7 at size 10 give 70 cards plus 1 on page 8 = 71; wont-fit gives 24 over 3 pages |
| Model stays in list when also a Top Pick | PASS | DeepSeek-Coder-V2-Lite-Instruct is Best quality pick and card 1 of Fits |
| Fits page 1 order | PASS | DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-V2-Lite, DeepSeek-V2-Lite-Chat, Qwen2.5 14B Instruct, DeepSeek-R1-Distill-Qwen-14B |
| Won't fit order, page 1 | PASS | Qwen3-30B-A3B 4.4, -Instruct-2507 4.4, Coder-30B 4.4, R1-Distill-32B 6.3, Qwen2.5-32B 6.3 |
| page=5 | PASS | V3.1 603.0, V3.2 603.8, V4-Pro 1439.2, V4-Pro-0813 1486.4 |
| page=4 | PASS | V4-Flash-0731 255.2, R1, V3, R1-0528, V3-0324 all 603.0 |
| 1 GB page 2 | PASS | Qwen3-0.6B-Base 0.5, Llama-3.2-1B 0.8, Llama-3.2-1B-Instruct 0.8, gemma-3-1b-it 0.8, phi-1_5 1.1 |
| quant_bucket=Q8 | PASS | page 1 starts Qwen3-30B-A3B 20.9 GB over (one Q8 pane per card) |
| Card contents R-015 to R-021 / Won't fit cards | PASS | each card: "x GB over", full gauge, "N GB needed", "24.0 GB", 4 buttons data-oversize and computed line-through, why-panel summary with hx-get |
| Required/available/margin on open and switched variant | PASS | Q4 28.44 needed, 24.0, 4.4 over; Q8 pane 44.94 needed, 24.0, 20.9 over |
| Quant press: no request, no list/count move | PASS | after pressing Q8 no request logged; counts stayed 71/0/24 |
| Back to card after paging opens default | PASS | Next then Previous: card 1 open on Q4 |
| Estimation unchanged | PASS | estimation.py, how_it_works.html unchanged vs main; ESTIMATION_POLICY_VERSION "v0.1" same as main |
| Mockups gist-list-tight / gist-list-wont-fit | PASS | Full-page screenshots `list-<tight|wont-fit>-<1280|390>-<light|dark>.png` here; identical dimensions to the mockups in all 8 cases; pixel diff (>24 levels) 1.3-1.6% light, 2.6-3.2% dark; no horizontal scroll (scrollWidth = viewport). Console: only a font preload warning, not caused by the change |
