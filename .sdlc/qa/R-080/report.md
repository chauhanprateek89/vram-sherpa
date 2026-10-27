# R-080 QA report (2026-10-27, branch sdlc/full-ui-overhaul)

App: http://localhost:8000, /healthz ready. Raw headers for all nine check URLs: `.sdlc/qa/R-080/curl-headers.txt`
(`curl -s -o /dev/null -D - '<base><url>'`, `-H 'HX-Request: true'` for URLs 3 and 4).

| # | URL | Status | Content-Type | Cache-Control (count) |
|---|-----|--------|--------------|-----------------------|
| 1 | / | 200 | text/html; charset=utf-8 | no-cache (1) |
| 2 | /results?... | 200 | text/html | no-cache (1) |
| 3 | /results?... + HX-Request | 200 | text/html | no-cache (1) |
| 4 | /results/partials/content?... + HX-Request | 200 | text/html | no-cache (1) |
| 5 | /results/partials/summary?... | 200 | text/html | no-cache (1) |
| 6 | /results/partials/list?... | 200 | text/html | no-cache (1) |
| 7 | /results/why/variant_model_qwen25_7b_instruct_q4?... | 200 | text/html | no-cache (1) |
| 8 | /models/model_qwen25_7b_instruct?... | 200 | text/html | no-cache (1) |
| 9 | /how-it-works | 200 | text/html | no-cache (1) |

Criteria
1. HTTP 200, text/html, exactly one Cache-Control: PASS (all nine).
2. Value contains no-cache or no-store: PASS (`no-cache`).
3. No immutable, max-age or s-maxage above 0: PASS (value is only `no-cache`).

Also: `pytest tests/test_cache_control.py` passes (27 tests, existing, added by the build). T-007 release.yml has the
GET / Cache-Control check with `curl -D -` and a no-cache|no-store grep.
Observation (not an R-080 criterion): GET / is about 617 KB and /results about 750 KB.

Result: PASS. T-001 and T-007 marked passes: true (T-007 also lists R-062 and R-068; orchestrator reconciles).
