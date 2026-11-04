# R-016 QA (T-036)

Criterion (R-086): model with no usable variant opens on recommended variant, else first in Q4,Q5,Q8,FP16 order.

PASS.
- `pytest tests/test_verdict_lists.py`: 10 passed, including test_default_none_usable_takes_recommended (Q5 recommended wins), test_default_none_usable_no_recommended_takes_first (Q4), test_no_usable_model_still_gets_a_card_under_wont_fit.
- Live: `curl "http://localhost:8000/results?vram_gb=1&verdict=wont-fit&page_size=10"` -> 200; every card (10) is a Won't fit model with aria-pressed="true" on Q4 only, others false (evidence: wontfit-1gb.html).
- Live with filters leaving only Q8/FP16 (`&quant_bucket=Q8&quant_bucket=FP16`): 200; every card opens on Q8 (first in order), FP16 false (q8.html).
- UI screenshots/mockup comparison not taken: the criterion is server-side card default selection, verified in markup.
