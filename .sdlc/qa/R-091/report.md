# R-091 QA report

All criteria PASS. Scripts in `scripts/`; raw output in `check.txt`, `interact.txt`, `a11y.txt`, `headring.txt`, `pixdiff.txt`. Screenshots: `gist|mid|wf10-<1440|390>-<light|dark>.png`, `focus-*-<1280|390>-<light|dark>.png`, `after-*.png`, `heading-after-next-*.png`. No console errors in any run.

- Group name and control names: PASS. `group "Show models by verdict"` with links `71 Fits`, `24 Won't fit` and inert `0 Tight` (a11y.txt).
- aria-current: PASS. Only the selected control has `aria-current="true"` (check.txt, 3 URLs, both themes).
- Non-colour selected mark, 3:1: PASS. 3 px solid border vs 1 px on others; light border rgb(20,17,43) on fill rgb(211,246,228), dark rgb(243,241,255) on rgb(11,48,35), far above 3:1.
- nav Pages: PASS. `navigation "Pages"`, links `Page <n>`, `…` `aria-hidden="true"`, current page a span `aria-current="page"` with tabindex -1.
- Keyboard: PASS. Tab order follows on-screen order (verdicts, picks, sizes, cards, Previous, numbers, Next); inert `0 Tight`, current page and `…` are not tabbable. Focus ring 3 px solid rgb(74,23,184) light, rgb(205,185,255) dark.
- Focus to heading after verdict, page and size loads: PASS. `document.activeElement` is `#list-heading` after Next, Page 1, size 10, Won't fit and gist to Fits, with body marker kept (in place, no navigation). Keyboard focus shows the ring on the heading; mouse focus shows none (:focus-visible).
- Heading into view: PASS. Heading top at 96 px (below sticky header) at 1440; at 390 at 0.
- Re-select selected verdict: PASS. Back on the gist in place (marker kept), focus on the `71 Fits` link, by keyboard and by mouse.
- Reduced motion: PASS. Scroll jumps 738 to 595 with no frames between; without it 643, 600, 595 (smooth).
- Modified clicks: PASS. Ctrl-click and Shift-click open a new tab with the full URL, no partial request, current page unchanged; middle click opens a tab. Meta-click on Linux is a normal navigation to the full URL (not a new-tab modifier on this OS), not an in-place swap.
- Tap targets at 390: PASS. No verdict, size, page number, Previous or Next under 44 x 44, no overlaps (check.txt `small: []`, `overlaps: []`).
- Three verdicts on one row at 390: PASS (same y: 873 / 846).
- scrollWidth 390 at 390: PASS for the gist, `&verdict=fits&page=6` (window `1 … 5 6 7 … 15`) and `vram_gb=1&verdict=wont-fit&page=10` (`1 … 9 10 11 … 19`), light and dark. Also 390 after in-place Fits and Next on a touch phone.
- Contrast, 12 px, non-colour cue: PASS. Smallest text 14 px on all 12 checked pages; verdict tiles keep icon shapes and labels; focus and selected marks above 3:1. Mockup pixel diff: focus-verdict 1280 0.12% light and dark; the other three mockups are viewport captures at an unknown scroll offset (17 to 34% diff); visual review of page 1280 light and heading 1280 dark shows the same layout, ring and content. Verdict 390 differs 9 to 19% (full-page capture vs mockup); visual layout matches (three tiles in a row, inert Tight dashed).
