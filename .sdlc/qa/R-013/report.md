# R-013, R-006, R-007, R-026 QA
- PASS /results with no hardware (also with other params, blank gpu_search/vram_gb): 302, Location: /, Cache-Control: no-cache (curl -D-).
- PASS /results?vram_gb=abc and ?gpu_id=no_such_gpu: 200, not redirected. HX-Request /results: 200.
- PASS / with query params renders landing, no results; no context_tokens control on / (count 0 at 1440 and 390).
- PASS chips on / are <a href> to /results (gpu_id / vram_gb=8 / vram_gb=24, context_tokens=2048); results show NVIDIA GeForce RTX 4060 (8.0 GB) catalog 2026-10-23, Manual VRAM (8 GB), Manual VRAM (24 GB); 2k selected; switching to 4k works (4096 in URL).
- PASS unseeded: tests/test_r083_landing.py::test_unseeded_landing and test_app.py unseeded tests pass (pytest 99 passed).
- PASS no console errors.
- Mockup (final-landing-empty): same content/layout; minor vertical spacing differences inside card (live card ~50px taller, larger gap above subtitle). Judged small. Screenshots: 1-landing-*.png, 2-chip-results-*.png, mock-landing-*.png.
