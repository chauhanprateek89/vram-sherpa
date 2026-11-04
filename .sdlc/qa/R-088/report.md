# R-088 QA report
All 17 criteria pass. Evidence: curl-params.txt, curl-misc.txt, py-links.txt, pixdiff.txt, PNGs in this folder.
- Query read on all routes: pass (200 on /results and 3 partials with HX-Request).
- Address/history: pass (gist -> Fits -> page=2 -> size 10; page omitted at 1, page_size omitted at 5; one entry each).
- Links carry other params in order (repeated family/quant_bucket): pass (py-links.txt).
- Unusable hardware (vram_gb=abc, gpu_id=no_such_gpu, gpu_search=RTX) + verdict link: stays on /results, Assumed VRAM 8 GB, same message: pass (unusable-*.png).
- Reload / fresh context: same heading and cards: pass.
- Back x2 / Forward x2 at 1280 and 390: page 1 Fits, gist, then restored: pass. History-miss variant covered by R-081 QA.
- /models/<id> then Back: same verdict, page, size: pass.
- No verdict: gist, page/page_size ignored (page=2&page_size=10 gives gist): pass.
- /results/partials/content normal load: 302 to /results?...family=llama&family=qwen&verdict=fits&page=2&page_size=10: pass.
- /results?verdict=fits no hardware, no HX: 302 to /; with HX 200: pass.
- Invalid verdict (green, FITS, empty): 200 gist. Invalid page_size (7, abc, 1000, empty): 200, 5 cards. Invalid page (abc,0,-1,1.5): 200 page 1. page=99 == page=15 (same cards, no redirect): pass.
- Repeated verdict: last wins (fits,tight -> Tight; tight,fits -> Fits): pass.
- verdict=<script>: no value reflected, no link carrying it, only app's own scripts: pass.
- No cookie set; localStorage holds only vramsherpa-theme; no console errors: pass.
- Mockup compare list-fits at 1280 and 390: same page size as mockup; light diff 0.1 percent; dark 2.3-3.6 percent (sticky header position in a full-page capture), pass.
