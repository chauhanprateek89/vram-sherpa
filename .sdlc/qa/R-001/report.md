# QA report R-001, R-002, R-003, R-005 (T-023, T-024, T-025, T-031): PASS

App at http://localhost:8000 (200). `pytest tests/test_r083_box_rule.py tests/test_gpu_input.py tests/test_r083_redirect.py`: 81 passed.
Evidence: `.sdlc/qa/R-001/` (1-typed12-1440, 2-results-390, 3-results-1440, 4-after-clear-1440, shot-*-light, cmp-*).

## R-003 box number on /results (curl, HTTP 200 each)
- gpu_search=12%20GB: `Manual VRAM (12 GB)`, vram_gb value 12, Search GPU value empty, data-box-vram=12, no note, no GPU message. Pass.
- 12.5gb: `Manual VRAM (12.5 GB)`, field 12.5. 8GB: `Manual VRAM (8 GB)`, field 8. 24: `Manual VRAM (24 GB)`, no "matched multiple". Pass.
- gpu_search=0: 200, `Assumed VRAM (8 GB)`, vram field 0 aria-invalid, default note, no GPU message; same as vram_gb=0. Pass.
- Regression: gpu_search=RTX gives the "matched multiple entries (...)" message; `RTX 3060 12GB` gives "was not found. Using 12 GB inferred from your input." Pass.

## T-025 in-place refresh (Playwright, 1440, `/results?vram_gb=24&context_tokens=4096`, VRAM emptied, Check compatibility)
- Typed `12`: field 12, Search GPU empty, `Manual VRAM (12 GB)`, no note; pushed URL keeps gpu_search=12; reload gives same. Pass.
- `12 GB`: same. `0`: field 0, aria-invalid, input-issues error, no GPU message, note shown. `RTX 4090`: GPU selected, field empty. Both emptied: `Assumed VRAM (8 GB)` plus note (R-005). Pass.
- Observation (not a failure): while the suggestion list is open the dropdown covers the submit button, so a click needed Escape first. Not part of this change.

## R-001 Clear (T-031) and R-002/R-005 on /
- `/results` Clear is `<a id="clear-gpu" class="btn-clear" href="/">Clear</a>`, 52 px tall, bordered, secondary look (same class as the chips/button rules). Click loads `/`: empty box, no Available VRAM field, no results, no note. Plain anchor, so works without JS. Pass.
- `/`: no vram_gb field, no results, no assumed-8 note. Pass. Unusable hardware cases (vram_gb=abc, gpu_id=no_such_gpu) give Assumed VRAM (8 GB) with note. Pass.
- Console: no errors (only a font preload warning).

## Mockups (docs/design/final, light, top of page; app pages are far longer than the sample mockups so only the first viewport is diffed, threshold 24)
- results-from-landing (gpu_search=12 GB): 1280 0.88%, 390 3.78%. results-manual-vram (vram_gb=24): 1.52%, 4.68%.
- results-errors (vram_gb=0): 1280 1.76%, 390 12.9%. Differences at 390: the app shows the real R-008 message ("vram_gb must be greater than or equal to 1e-06.") and value 0 where the mockup has "Invalid value for vram_gb." and an empty field, and Top Picks is open in the app only because the mockup is collapsed there. Layout and elements match. Judged not a failure (existing R-008 behaviour, unchanged by R-083).
