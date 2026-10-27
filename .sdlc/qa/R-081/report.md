# R-081 QA report (Playwright, Chromium, 390 and 1280 px; app at http://localhost:8000)

Result: all criteria PASS. T-002 and T-003 marked `passes: true`.

Evidence: curl output in `curl-evidence.txt`; screenshots in this directory (`*-1280.png`, `*-390.png`); `pytest tests/test_history.py` 35 passed.
A history miss was forced with `localStorage.removeItem('htmx-history-cache')`. "Full page" = exactly 1 header, 1 `#search-form`, 1 footer, site stylesheet applied.

## Back and forward
- Steps A, miss, Back: PASS (both widths). Full page, `/results` path, 8k preset checked, summary "Context: 8k", cards identical to `/results?vram_gb=24&context_tokens=8192` opened directly. `back-miss-8k-*.png`.
- Miss, Forward: PASS. 2k preset checked, summary 2k, cards identical to direct 2k page. `forward-miss-2k-*.png`.
- No nested header/form/footer: PASS (counts 1/1/1 in every state).
- Preset change after a miss-restored page: PASS. A marker set on `window` survived (no full page load) and the results refreshed to 4k.
- Back/Forward with cache untouched: PASS for "full page for that entry" (1/1/1, URL and summary right). See observation.
- 12 refreshes (2k/8k alternating) then 12 Backs: PASS at both widths, with the cache left alone and with a miss forced before every Back (12 full-page navigations). Every press showed the full page with the right context. `twelve-backs-*.png`.

## Address bar, reload, shared links
- Address after an in-place refresh: PASS. Path `/results`, query `?gpu_id=&gpu_search=&vram_gb=24&context_tokens=2048&min_params_b=&max_params_b=` (contains `vram_gb=24`, `context_tokens=2048`).
- Steps A then reload: PASS. Full page, 2k checked, summary 2k, state signature identical before and after (models, order, verdicts, figures).
- Steps B then reload: PASS. qwen and llama ticked, 37 cards, signature identical.
- Steps B, address opened in a new isolated browser context: PASS. Full page, qwen and llama ticked, 37 cards, same signature, stylesheet applied. `stepsB-private-*.png`.

## Old-form addresses (curl-evidence.txt)
- No `HX-Request`: 302, empty body, `Location: /results?vram_gb=24&context_tokens=4096&family=qwen&family=llama&quant_bucket=Q4&quant_bucket=Q8` (byte for byte, order kept): PASS. `HX-Request: false` also 302.
- No query: `Location: /results`: PASS.
- Location is a path only (no scheme or host); status is 302, never 301 or 308: PASS. It also sends `Cache-Control: no-cache` and `Vary: HX-Request, HX-History-Restore-Request`.
- Browser opens `/results/partials/content?...family=qwen&family=llama`: PASS. Chain 302 then 200, ends at `/results?vram_gb=24&context_tokens=4096&family=qwen&family=llama`, full page, signature identical to the direct `/results` address.
- Old-form address then 8k preset: PASS. In place (window marker survived), 8k checked, qwen and llama still ticked. `oldform-then-8k-*.png`.

## Related (T-002/T-003 acceptance and R-032)
- `HX-Request: true` on `/results/partials/content?...family=a&family=b`: 200 fragment, `HX-Push-Url: /results?vram_gb=24&context_tokens=4096&family=a&family=b`. No query gives `HX-Push-Url: /results`: PASS.
- `/results` with `HX-Request` plus `HX-History-Restore-Request`: full page (3 matches of header, form and footer). `HX-Request` only: fragment (0 matches): PASS.
- `<meta name="htmx-config" content='{"refreshOnHistoryMiss":true}'>` at line 36, before the htmx script at line 37: PASS.
- `/results/partials/summary` and `/list` return 200 and do not redirect: PASS.
- `/results/partials/content?context_tokens=abc` returns 422 JSON, the same as `/results?context_tokens=abc`: PASS.
- Unknown `Host` returns 400 `{"detail":"Host not allowed"}` with no `Location`: PASS.
- Opening a breakdown ("How this is estimated") leaves the address and history length unchanged (49 before and after): PASS.
- Mockup (`docs/design/final/results-manual-vram-1280-light.png` and `-390-light.png`): the restored and reloaded pages at 4k match the layout and content at both widths (`mockup-compare-top-390.png`, `mockup-compare-top-1280.png`, `back-miss-8k-1280.png`). The only differences are the context-dependent figures at 8k.
- Console: no errors. Only a repeated warning that the `outfit-v1-latin-wght.woff2` preload was unused (not caused by R-081).

## Observation (not a failure)
When Back or Forward is served from the htmx cache (no miss), the summary and results are right for that entry, but the context preset radio can show the preset the page loaded with (for example 4k checked while the summary says 8k). This is htmx 1.9 snapshotting markup rather than live control state. R-081 requires the preset to be selected only after a history miss, and that holds. Worth a follow-up if the user wants cache hits to match.
