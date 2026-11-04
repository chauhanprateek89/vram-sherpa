# R-022 QA
Criterion (Changed R-084, R-085): the gist states the R-084 note, a selected verdict states the R-085 note. PASS.
Evidence (curl and Playwright at 1440 and 390, screenshots `.sdlc/qa/R-022/{gist,fits,zero,empty,wont}-{1440,390}.png`):
- Gist (`/results?vram_gb=24&context_tokens=4096`), zero-count verdict (`verdict=tight`) and empty (`vram_gb=1`): `Choose Fits, Tight or Won't fit to see the models.` once, below the verdict controls.
- `verdict=fits`, `verdict=wont-fit`: `Pick a quantisation on any card to see how it changes.` once, below the controls; no gist note.
- `Only Fits and Tight models are listed` / `Showing only` absent in every state.
- `Summary for this query`, `Hardware:`, `Context:` present. No console errors.
Pixel diff against mockups not run; the criterion is text and placement.
