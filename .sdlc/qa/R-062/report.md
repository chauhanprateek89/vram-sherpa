# R-062 QA (T-051)
Mockups: not applicable (CI workflow change, no UI).

- PASS: release.yml step "Smoke test the published image" holds the verdict check: curl -sS -m 10 (no -f, no -L), status must be exactly 200 and body must contain `data-model-card` (grep -qF), else `::error::` and `exit 1`, before the tag step.
- PASS: same request against local app http://localhost:8000/results?vram_gb=24&context_tokens=4096&verdict=fits -> HTTP 200, 5 occurrences of `data-model-card` (first: `<article class="model-card inset" data-model-card ...>`).
- Negative controls: no verdict -> 200, 0 cards (gist); verdict=nope -> 200, 0 cards. So the check does discriminate.
- PASS: `pytest tests/test_r082_release_order.py` -> 7 passed.

Caveat: the workflow itself first runs for real on merge; it was not executed here. Only the request was run against the local app.
