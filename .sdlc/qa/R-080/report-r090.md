# R-080 QA (2026-11-04, URLs 1-13)
Raw headers: .sdlc/qa/R-080/curl-headers-r090.txt (curl -s -o /dev/null -D -, HX-Request on 3, 4, 12).
1. 200, text/html; charset=utf-8, exactly one Cache-Control on all 13 URLs: PASS
2. Value is `no-cache`: PASS
3. No immutable, max-age, s-maxage: PASS
pytest tests/test_cache_control.py: all passed.
