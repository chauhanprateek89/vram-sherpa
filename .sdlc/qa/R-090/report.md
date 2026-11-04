# R-090 QA report

All criteria pass. Full pytest suite passes (run at start). Scripts in `scripts/`.

- In-place refresh (verdict, page, size) via `/results/partials/content`, no page load, typed text kept: PASS. Marker survived; field kept `12` typed; requests `.../partials/content?vram_gb=24&context_tokens=4096&verdict=fits&page=2` and `...&page_size=10`. See 1-loading-from-gist-1280.png, 2-after-fits-1280.png.
- Typed text not used: PASS (request carries address vram_gb=24, not 12).
- Loading: PASS. During a held response `#results-shell` has aria-busy=true, `Loading models` shown, old view kept (pixdiff.txt: 0.32% vs mockup loading-from-gist at 1280).
- Failure (HTTP 500 and network abort): PASS. `role=alert` shows `Could not load the models. Try again.`, URL and cards unchanged, aria-busy removed, alert cleared on next successful request. Mockup diff failed-from-gist 1280: 0.30%. 3-failed-next-page-1280.png.
- partials/list without verdict: 200, 0 cards. With verdict: 200, cards. partials/summary: 200, answer tile, no cards, aria-current set when verdict=fits. api.txt.
- No JS: PASS. Verdict controls, sizes, page numbers, Previous/Next are plain links; Next/size/verdict links load full URLs; Apply filters keeps verdict (apply.txt); Top Picks is a native details (nojs.txt, 5-nojs-list-*.png).
- Cache-Control: `no-cache`, single header, 200 text/html on the checked URLs (api.txt).
- Mockup comparison: 1280 loading-from-gist and failed-from-gist match (0.3%). Other states differ 11-29% in pixdiff.txt only because the mockups are viewport captures at an unknown scroll offset; visual review of the 1280/390 light pairs shows same layout, pill, ring on pressed control and spinner on Next/Fits. 390 gist states differ in scroll offset only.
- Pending page-size link (`.sizes-seg .is-pending`, no mockup): 6-pending-pagesize-*-{light,dark}.png, crops `6-pending-pagesize-crop-*`. Shows a small spinner arc at the right of the `10` label inside the pill; at 390 the arc sits tight against the `0` (touching). Cosmetic, not a failure.
- Console errors: none from the app (only from injected 500/abort).
