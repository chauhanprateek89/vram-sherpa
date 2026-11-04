# R-023 QA report

Tasks T-036, T-039, T-040. App at http://localhost:8000. Script: `.sdlc/qa/R-023/check_counts.py`, output `counts.log`.

1. Fits + Tight = headline N: PASS. 8 queries, e.g. vram_gb=8: 49+9=58 headline 58; vram_gb=24&ctx=4096: 71+0=71; vram_gb=12: 62+1=63; llama family: 6+4=10; vram_gb=1: 0=0.
2. Each count equals the total of that verdict's list across all pages: PASS. Walked every page at sizes default, 5, 10, 25 for fits, tight, wont-fit in all 8 queries; card totals matched the counts (e.g. 49/9/37).
3. Counts unchanged by verdict, page, page size: PASS. Every walked page showed counts identical to the gist's.

Console (chrome devtools) on the tight list: no errors or warnings.
Mockup comparison (docs/design/final/gist-*): structure and layout match at 1280 and 390, light and dark (screenshots `*-<w>-<scheme>.png`; diffs 3-15 percent, from differing data since mockups use other GPUs for some states, pagination height). No missing element.
