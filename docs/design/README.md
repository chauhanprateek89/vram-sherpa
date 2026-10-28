# Centred landing box: design (Gate 2 for R-083)

`/` becomes one large box in the middle of the page, like a new chat in Claude. Type a GPU, or a VRAM number such as `12 GB`, and you go to the results page; `Clear` there brings you back. Same direction as the overhaul ("D layout + E glass", Outfit). The overhaul section further down is unchanged and still describes every other screen.

**Gallery (private Artifact, new page):** https://claude.ai/artifact/S8yurugqYYCy6CNtbNTSbj
It holds today's `/` beside the new one, every state at 390 and 1280 px in light and dark, the wording, the measurements and the decisions. Local copy: `docs/design/landing-gallery.html` (open it from this folder).

## States (R-067: L1 to L4)

Each has 4 renders (390 and 1280 px, light and dark). Header and footer are the ones of every page (R-076). Names and labels come from the running app (catalog 2026-10-23).

| Mockup (`docs/design/mockups/final-<name>.html`) | State | R-067 | Requirements |
|---|---|---|---|
| `landing-empty` | `/`, empty box, focus in the box as it loads | L1 | R-083, R-067, R-077 |
| `landing-suggest` | Styled suggestion list open after typing `4060` (replaces `gpu-suggest` as the reference for `/`) | L2 | R-071, R-083 |
| `landing-unseeded` | Catalog unseeded: box, chips, then the two R-026 sentences in a status notice; inert `Example GPU` chip | L3 | R-026, R-083 |
| `landing-focus-box` | Focus on the box, `12 GB` typed | L4 | R-077, R-083 |
| `landing-focus-chip` | Focus on a chip | L4 | R-077, R-083 |
| `landing-focus-submit` | Focus on `Check compatibility` | L4 | R-077, R-083 |
| `results-from-landing` | Where `12 GB` leads (`/results?vram_gb=12&context_tokens=2048`); `Clear` is a link to `/` | (results) | R-001, R-003, R-013, R-083 |

Renders: `docs/design/final/<name>-<390 or 1280>-<light or dark>.png`. Today's `/` for comparison: `docs/design/before-landing/home-<390 or 1280>-<light or dark>.png`.

Rebuild: `build/extract_landing.py` (app up) saves the results page; `build/make_landing.py` writes the mockups from `build/landing.css`, `components.css` and `tokens.css`; `node build/render.mjs landing` (and `results-from-landing`) renders them; `node build/measure_landing.mjs` prints the numbers below; `build/make_landing_gallery.py` writes the gallery. Playwright needs Node 20 or later.

### What was built to the HLD

The structure follows `docs/HLD.md` ("Centred landing box") and ADR-0009 to 0013: one `h1`, the help line, `form#landing-form` (`method="get" action="/results"`) holding a visually hidden label `Search GPU`, the box `#gpu_search` (inside `.combo`, `autofocus`, `data-submit-on-choose`) and one `button type="submit"` named `Check compatibility`; then `role="group" aria-label="Examples"` holding three `a.chip` links; then, when unseeded, a `role="status"` element. No `Clear`, context control, filters or second text field. The styled list is the existing `.listbox` (`role="listbox"` under the box, 48 px options). `Clear` on `/results` is `a#clear-gpu.btn-clear` with the secondary-button look. The landing styles are in `docs/design/landing.css`, inlined in every landing mockup (to be appended to `styles.css`; `build/` is not tracked).

## Results pages checked

- **Changed:** only `Clear`, which is a link to `/` drawn as the same secondary button (`results-from-landing` shows it). No other look changes.
- **Checked, unchanged:** the other 22 approved states (`results-gpu`, `results-manual-vram`, `results-gpu-ambiguous`, `results-gpu-notfound`, `results-errors`, `results-chips`, `results-filters-open`, `results-picks-hidden`, `results-tight-gated`, `why-closed`, `why-loading`, `why-loaded`, `empty-links`, `empty-inert`, `empty-mixed`, `unseeded`, `detail-gated`, `detail-ungated`, `detail-no-sources`, `detail-error`, `how-it-works`, `focus-states`). Accepting a number in `Search GPU` does not change the `/results` help line, placeholder or any message: the `Available VRAM (GB)` field is still there and the line "Pick your GPU, or enter your VRAM if it isn't listed." is still true. A box number of `0` shows the `results-errors` look with the R-008 message for 0. In those mockups the logo, `Estimator` and `Clear` still point at the old home mockup; that is a link target, not a look.
- **Superseded for `/`:** `home-default` (state 1) and `gpu-suggest` (on `/`). They stay in the repo for `/results`.

## Copy changes

Three strings, all rewordable (R-078); each is also a row in the copy table below. The accessible names `Search GPU` and `Check compatibility`, the chips, `Catalog is not initialized yet.`, `Seed the catalog data, then refresh this page.` and the footer stay word for word.

| Where | Old | New |
|---|---|---|
| Landing `h1` | Find models that fit your VRAM | Which local LLMs fit your GPU? |
| Landing help line | Pick your GPU, or enter your VRAM if it isn't listed. | Type a GPU name, or enter its VRAM, like 12 GB. |
| Landing box placeholder | Start typing a GPU name | Your GPU, or its VRAM |

Removed from `/` (R-083, Q-17): the summary and headline, Top Picks, cards, active filter chips, the empty state, the filters, `Available VRAM (GB)`, the context window and its presets, the assumed 8 GB note, messages and errors, and `Clear`. All stay on `/results`.

## Measurements QA can take (Chromium, light and dark alike)

| Check | Mockup |
|---|---|
| Box width (`#gpu_search`) | 344 px at 390 (needs 342); 754 px at 1280 (needs 560). At 390 the pane has 6 px of padding; more padding or a thicker border breaks the 342 |
| Box centre vs viewport centre | 0.00 px at both widths (needs within 2) |
| Box vertical centre | L1 and L2: 416 px of 844 (49.2%) and 428 of 800 (53.5%). L3: 360 (42.6%) and 382 (47.8%). Needs 35 to 65% |
| Box, chips, submit without scrolling | Chips end at 510 of 844 and 532 of 800; each state is exactly the viewport tall |
| Tap targets | Submit 48 x 48 at 390, 232 x 60 at 1280; chips 44 px tall; options 48 px |
| Contrast | Lowest text 7.07:1 light, 8.28:1 dark; 0 failures in 994 text boxes over 28 renders (the hidden 1 px submit label at 390 is excluded) |
| Focus | Box focused on load; 3 px solid ring on box, chip and submit |
| Inputs | 1 text input, 1 `h1` on `/` |

## Decisions for you

1. **Wording** of the greeting, help line and placeholder (table above). An alternative greeting is the one from your 24 October prototype, `What are you running it on?`; I picked one that names the product's job so a first-time visitor needs no other context.
2. **`Example GPU` chip when the catalog is unseeded.** Drawn dashed and inert, as ADR-0009 says, so the row keeps three chips. A dead chip invites a tap that does nothing; the alternative is two chips in L3 only.
3. **Submit control.** Words (`Check compatibility`) beside the arrow at desktop width, arrow only on a phone; the accessible name is the same.

## Notes and conflicts

- No conflict with `docs/HLD.md` or ADR-0009 to 0013. The order in the unseeded state (box, chips, notice) follows ADR-0009's content order.
- The greeting sits inside the same glass pane as the box because on the bare background its contrast would be about 2.2:1 in the dark theme at the worst backdrop colour (R-077 needs 4.5:1).
- `Clear` as a link is announced as a link named `Clear`, not a button (ADR-0012).
- `autofocus` (R-083) opens the on-screen keyboard at once on Android Chrome, covering the chips; iOS Safari usually ignores it.
- The styled-list pick and chips navigate in the live mockups to the results mockup; the mockups do not call the server.

---

# UI/UX overhaul: design (Gate 2, round 2 of 2)

The direction you picked is **D layout + E glass**: D Bento's tile grid, built from E Glass's frosted panes, inset sheets, Outfit type, shadows and motion. This round mocks up every screen and state of R-067 in it, plus the extra states R-073 to R-077 need.

**Gallery (private Artifact, republished to the same link):** https://claude.ai/artifact/GpFFFNrx4duQTg63MYdQQQ
It holds the before screenshots, every state at 390 and 1280 px in light and dark, the live mockups (hover, press and focus work), the spec sheet and the copy table. Local copy: `docs/design/gallery.html` (open it from this folder).

## What you are asked to approve

1. The look and layout of every state below.
2. **The copy table** (Q-2) further down: every string that changed, old to new, and where it appears. Errors, verdict labels, figures, field labels, button labels, link text and tags are word for word (R-078); the second list under the table says so string by string.
3. Three decisions, listed under "Decisions for you".

## Files

| What | Where |
|---|---|
| Tokens (colour, type, space, radii, shadows, glass, motion) | `docs/design/tokens.css`, linked from every mockup |
| Component styles (what becomes `styles.css`) | `docs/design/build/components.css`, inlined into every mockup |
| Mockups (self-contained HTML, hover, press and focus live) | `docs/design/mockups/final-<name>.html` (`?theme=dark` for dark) |
| Renders, 4 per mockup | `docs/design/final/<name>-<390 or 1280>-<light or dark>.png` |
| Spec sheet | `docs/design/mockups/final-spec.html` (a document, open it in a browser; add `?theme=dark` for dark) |
| Logo mark and favicon | `docs/design/marks/final-mark.svg`, `final-favicon.svg` |
| Font (Outfit, SIL OFL 1.1, one woff2) | `docs/design/fonts/outfit-latin-wght-normal.woff2` |
| Today's app, for comparison | `docs/design/before/results-<390 or 1280>-<light or dark>.png` |
| Round 1 boards D, E, F (not selected) | `docs/design/rejected-r1b/` (A, B, C: `docs/design/rejected-r1/`) |
| Rebuild everything | `docs/design/build/` (`extract.py` reads the running app, `make.py`, `make_spec.py`, `render.mjs`, `contrast.mjs`, `make_docs.py`) |

## Screens and states

Every row has 4 renders (390 and 1280 px, light and dark). The header and footer (R-076) are on every page, identical. All figures, names and messages come from the running app (catalog 2026-10-23); the mockup queries are in the last column. Lists are cut to the first 4 to 8 cards.

| Mockup | State | R-067 state | Requirements | Built from |
|---|---|---|---|---|
| `home-default` | Home, no input | 1 | R-005, R-013, R-068 to R-071, R-076 | `/` |
| `results-gpu` | A GPU selected | 2 | R-001, R-004, R-070, R-072 | `/results?gpu_id=gpu_nvidia_rtx_4090_laptop_16gb&context_tokens=4096` |
| `results-manual-vram` | Manual VRAM: the reference query | 3, 8 | R-014 to R-024, R-069, R-072, R-076 | `/results?vram_gb=24&context_tokens=4096` |
| `results-gpu-ambiguous` | Ambiguous GPU search | 4 | R-003, R-005, R-022, R-071 | `/results?gpu_search=RTX` |
| `results-gpu-notfound` | GPU not found, with a GB hint | 5 | R-003, R-022, R-071 | `/results?gpu_search=RTX 3060 12GB` |
| `results-errors` | Inline input errors | 6 | R-008, R-071, R-077 | `/results?vram_gb=abc` |
| `results-chips` | Active filter chips | 7 | R-031, R-069 | `/results?vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4` |
| `results-filters-open` | Filters open | 7 | R-027 to R-030, R-069, R-078 | `/results?vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4` |
| `results-picks-hidden` | Top Picks hidden | 8 | R-024 | `/results?vram_gb=2&context_tokens=4096` |
| `results-tight-gated` | A Tight card, struck-through quantisations, Gated | 9 | R-016, R-017, R-021, R-068, R-072, R-075 | `/results?vram_gb=8&context_tokens=2048&family=llama&min_params_b=8&max_params_b=8` |
| `gpu-suggest` | GPU suggestions (optional replacement of the native list) | - | R-001, R-071, ADR-0005, Q-8 | `/ (typing `4060`)` |
| `why-closed` | Breakdown before it loads | 10 | R-020, R-073 | `/results?vram_gb=24&context_tokens=4096` |
| `why-loading` | Breakdown loading | 10 | R-073 | `/results/why/<variant_id>?...` |
| `why-loaded` | Breakdown loaded | 10 | R-020, R-073, R-075 | `/results/why/variant_model_deepseek_coder_v2_lite_instruct_q4?context_tokens=4096&vram_gb=24` |
| `empty-links` | Empty state, both next steps are links | 11 | R-025, R-069 | `/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8` |
| `empty-inert` | Empty state, both next steps are inert | 11 | R-025, R-069 | `/results?vram_gb=1&context_tokens=2048&quant_bucket=Q4` |
| `empty-mixed` | Empty state, one link and one inert | 11 | R-025 | `/results?vram_gb=1&context_tokens=4096&quant_bucket=Q4` |
| `unseeded` | Unseeded catalog | 12 | R-026, R-038 | `(empty database)` |
| `detail-gated` | Gated model with sources | 13 | R-033 to R-036, R-068, R-074, R-075 | `/models/model_llama_31_8b_instruct?vram_gb=24&context_tokens=2048` |
| `detail-ungated` | Ungated model with sources | 13 | R-033 to R-036, R-074 | `/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096` |
| `detail-no-sources` | Model without sources | 13 | R-033, R-074 | `/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096 (sources removed)` |
| `detail-error` | Model detail with an inline VRAM error | 13 | R-035, R-008, R-074 | `/models/model_qwen25_7b_instruct?vram_gb=abc` |
| `how-it-works` | How it works | 14 | R-037, R-075, R-076 | `/how-it-works` |
| `focus-states` | Focus ring on every control | - | R-077 | `(gallery of controls)` |

## Requirement coverage

| Requirement | Where it is designed |
|---|---|
| R-067 states 1 to 14 | Table above |
| R-068 one identity, logo, favicon | Every page; the logo in the spec sheet; one verdict pill and one gated tag used on cards and on model detail |
| R-069 answer first, density | Headline first in the results area. Reference query: headline ends at 167 px (1280) and 804 px (390); first card's name starts at 624 and 1154 px; its verdict and margin end at **770 px** (1280) and **1365 px** (390). Limits are 800 and 1688. Filters collapsed at both widths with `Filters (N)`; Top Picks tiles under the answer, one collapsed row on a phone |
| R-070 phone | `scrollWidth` is 390 on every mockup; no tap target under 44 px at 390; no text under 14 px; fields 16 px; the four quantisation buttons share one row; model detail rows stack with no horizontal scroll; header takes two rows |
| R-071 form | One error list and one GPU message, both inside the answer tile, so a refresh replaces them. Help text merged to one line. Optional suggestion list: see decisions |
| R-072 card, switcher, gauge | Verdict by label plus icon shape (circle tick, triangle bar, square cross); reserve as a dashed outline; Won't fit buttons struck through |
| R-073 breakdown | Closed, loading and loaded states |
| R-074 model detail | Four states; first variant's status ends at 657 / 1174 px (1280 / 390) for the ungated model, 744 / 1394 for the gated one; limit at 390 is 1688 |
| R-075 how it works | Ten terms, formulas unchanged, worked example 4.95, 0.98, 1.40, 7.33 GB with the figures from the requirement |
| R-076 header and footer | Same on every mockup; logo links home; three footer values stacked on a phone |
| R-077 accessibility | Contrast: rendered check over 9,522 text boxes in 96 renders, **0 failures**, lowest 6.29:1 (light) and 4.78:1 (dark); 86 token pairs including non-text, 0 failures. Focus ring on every control (`final-focus-states`). Reduced motion and reduced transparency fallbacks in the spec sheet |
| R-078 behaviour kept | Same fields, controls, ids and routes; wording rules in the copy table |

## Decisions for you

1. **Logo.** I merged the two: D's tile arrangement drawn in E's glass and gradients (a violet tile, a tall cyan tile and a wide outlined bar that is the fit gauge). It is neither D's four tiles nor E's two squares. Say if you would rather have one of the originals.
2. **GPU suggestion list (Q-8, ADR-0005).** I could not test iOS Safari or Android Chrome, so I have no evidence the native list works badly on phones. The mockups assume the native list stays (the default in ADR-0005). `final-gpu-suggest` shows the styled replacement in case you want it. If you approve it, ADR-0005 becomes accepted and the build adds the combobox.
3. **Filters inside the form tile, closed on desktop too.** R-077 wants tab order to match the screen, and the filters have to stay outside the htmx-swapped results (ADR-0006), so they sit at the foot of the form tile, not in a full-width row. ADR-0006 says desktop opens them when JavaScript runs; open, they push the first card's verdict to past 1,100 px at 1280, past R-069's 800, so the mockups keep them closed. Say if you would rather open them and relax R-069.

## Where a requirement pushed back

- **R-069 on messages.** Only the reference query is held to 800 px. Where an extra message shows, the first verdict ends below 800 px at 1280: assumed 8 GB note 798 (inside), ambiguous GPU 843, input errors 883. At 390 px every state is under 1534 px.
- **R-077 on glass.** Dark glass at 70% opacity measured 3.64:1 for muted text on the brightest backdrop colour, so the dark pane is 82% (round 1 finding). Phone panes without blur use an 86% fill; both are in the contrast table.
- **Glass budget.** At most 3 blurred layers on a phone (header, form, answer). Top Picks, the results pane and the footer are lite there. Wide: 6, and the cards are inset sheets in one pane, so 71 cards cost one blur. Solid fallbacks for no `backdrop-filter` and `prefers-reduced-transparency`.
- **R-020 order.** The breakdown lists the total before the reserve. The reserve is held back and is not added into the total (R-009, R-010), so between the addends it read like a fourth one. All five figures are unchanged.
- **R-021 and R-036.** The gated tag is styled the same on both pages (R-068) but the words stay `Gated` on a card and `gated` on model detail, as those requirements give them.
- **R-034.** The detail table keeps its seven columns and its sort (FP16, Q4, Q5, Q8) at 1280 and stacks each row at 390.
- **No model lacks sources.** `final-detail-no-sources` is hypothetical, as noted on its tile.

## Copy table (Q-2): what changed

Old is what the app renders today; new is what the mockups show; where says which tile. `N`, model names and figures come from the data.

| Group | Old | New | Where |
|---|---|---|---|
| Form | Choose either your exact GPU or enter available VRAM directly. | Pick your GPU, or enter your VRAM if it isn't listed. | Form tile on `/` and `/results`, under the title. Merges the next two rows. |
| Form | I know my GPU | (removed; the field's own label `Search GPU` names it) | Form tile, above `Search GPU`. |
| Form | If your GPU is not listed, leave this blank and enter VRAM instead. | (merged into the line above) | Form tile, under the GPU field. |
| Form | I know my VRAM | (removed; the field's own label `Available VRAM (GB)` names it) | Form tile, above `Available VRAM (GB)`. |
| Form | Start typing a GPU name (or leave blank)  (placeholder) | Start typing a GPU name  (placeholder) | `Search GPU` field. The old placeholder is the one the mockups can show in full at 390 px next to `Clear`. |
| Form | How it works  (a link inside the form) | (removed; the header already has the same link) | Form tile, under `Check compatibility`. |
| Form | Filters | Filters (N)  where N is the number of active filter chips; `Filters (0)` with none | Disclosure at the foot of the form tile, both widths (R-069). |
| Filters | Lower bound on model size in billions of parameters.  (help line) | (removed; the `title` tooltip on the field keeps the wording) | Filters disclosure, under `Min params (B)` / `Max params (B)`. |
| Filters | Upper bound on model size in billions of parameters.  (help line) | (removed; the `title` tooltip on the field keeps the wording) | Filters disclosure, under `Min params (B)` / `Max params (B)`. |
| Answer | Summary for this query  (heading) | 71 models fit or are tight  (the headline is the heading; N and the plural come from the data) | Answer tile, top. The `Hardware:` and `Context:` lines under it are unchanged. |
| Answer | Results  (heading) | Results  (kept for screen readers only, visually hidden) | Above the cards. |
| Answer | 71 models fit or are tight. Pick a quantisation on any card to see how it changes. | headline `71 models fit or are tight`, then the note below | Moved to the top of the results area (R-069). Singular stays `1 model fit or are tight` as today. |
| Answer | Showing only Fits and Tight models in the list below. | Only Fits and Tight models are listed. Pick a quantisation on any card to see how it changes. | Answer tile, note under the counts. The hover text (`Fits leaves reserve headroom. Tight can run, but is close to your VRAM limit.`) is unchanged. On the empty state only the first sentence shows. |
| Top Picks | Highest-ranked variant that remains in the Fits band. | The strongest model that still leaves room to spare. | `Best quality that fits` pick (R-069). |
| Top Picks | Next highest-ranked practical option by current ranking order. | Nearly as capable, and comfortable to run. | `Best balance` pick (R-069). |
| Top Picks | Lowest-ranked viable option, usually the lightest footprint. | The lightest model that still runs on your GPU. | `Best small` pick (R-069). |
| Empty state | There are 95 variants in Won't fit, hidden by default. | There are 95 models in Won't fit, hidden by default.  (`There is 1 model in Won't fit, hidden by default.` for one) | Empty state, under the two next-step links (R-069, Q-10). |
| Landing (R-083) | Find models that fit your VRAM  (the `h1` on `/`) | Which local LLMs fit your GPU? | `/` only, centred above the box. `/results` keeps `Find models that fit your VRAM` as its form title. |
| Landing (R-083) | Pick your GPU, or enter your VRAM if it isn't listed.  (help line on `/`) | Type a GPU name, or enter its VRAM, like 12 GB. | `/` only, under the `h1`. `/results` keeps the old line. |
| Landing (R-083) | Start typing a GPU name  (placeholder on `/`) | Your GPU, or its VRAM | `/` only, in the box. `/results` keeps the old placeholder. |
| Breakdown | weights_gb: 9.72 | Model weights   9.72 GB | How this is estimated, row 1. The figure is unchanged; the unit GB is added. |
| Breakdown | kv_cache_gb: 3.85 | Context memory (KV cache)   3.85 GB | Row 2. |
| Breakdown | runtime_overhead_gb: 1.79 | Runtime overhead   1.79 GB | Row 3. |
| Breakdown | required_vram_gb: 15.36  (last row) | Total needed   15.36 GB  (now the fourth row, ruled off from the three it adds up) | Row 4. It equals the card's `15.36 GB needed`. |
| Breakdown | reserve_gb: 2.40  (fourth row) | Safety reserve   2.40 GB, with the line `Kept free, not part of the total`  (now the last row, dashed) | Row 5. Moved after the total because the reserve is not added into it (R-009, R-010); between the rows it read like a fourth addend. |
| Breakdown | (nothing while loading; the placeholder `Open to load the breakdown.` shows) | Loading the breakdown  (with five shimmering rows) | While the request runs. The placeholder `Open to load the breakdown.` stays as it is for the moment before the request starts. |
| Model detail | Family: llama | Params: 8.0B | Type: instruct | License: Llama 3.1 Community License | four labelled tiles: Family llama, Params 8.0B, Type instruct, License Llama 3.1 Community License | Model tile. Same facts and values; the `Family:`, `Params:` pipes and colons become tile labels. |
| Model detail | (no heading over the recompute form) | Recompute for other hardware  (visually hidden heading) | Recompute tile; screen readers only. |
| Model detail | (no heading over the variant table) | Variants  (visually hidden heading) | Variant table; screen readers only. |
| How it works | Inputs: three bullets (`available_vram_gb` from GPU selection or manual VRAM input; `context_tokens` from presets (2048 default, 4096, 8192); variant and model metadata: `params_b`, `bits_effective`, `kv_gb_per_1k_ctx`) | five term tiles, each with a plain-language line and the old source line underneath | Inputs section. Plain-language lines: available_vram_gb: `The VRAM you have, from the GPU you pick or the number you type.` context_tokens: `How much text the model keeps in view at once: 2048, 4096 or 8192 tokens, shown as 2k, 4k and 8k.` params_b: `Model size in billions of parameters. A bigger model has more weights to store.` bits_effective: `Bits stored per parameter after quantisation: Q4 is 4.5, Q5 is 5.5, Q8 is 8.5, FP16 is 16. Fewer bits, less memory.` kv_gb_per_1k_ctx: `Memory the model's context (its KV cache) takes for every 1,000 tokens, in GB. Each model page shows this value.` The old source lines stay: `From GPU selection or manual VRAM input`, `From presets (2048 default, 4096, 8192)`, `Variant and model metadata`. |
| How it works | Formula: four bare formulas | the same four formulas, each under its term with a plain-language line | Formula section. weights_gb: `Memory for the model itself. Parameters times bits per parameter, divided by 8, gives gigabytes; the 1.10 adds 10% on top.` kv_cache_gb: `Memory for the context window you chose. It grows with the context length.` runtime_overhead_gb: `Memory the runtime needs besides the model: a flat 1.0 GB plus 0.05 GB for every billion parameters.` required_vram_gb: `Everything the variant needs: weights, KV cache and runtime overhead added up.` The formula text itself is unchanged. |
| How it works | Reserve buffer: the formula only | the formula, a plain-language line, and a 10 GB example | Reserve buffer section. reserve_gb: `Free space to leave as a safety margin: 10% of your VRAM, never less than 0.75 GB.` Example lines: `With 4 GB the reserve is 0.75 GB. With 16 GB it is 1.6 GB.` and, with a drawn gauge, `The reserve is 1.0 GB. Up to 9.0 GB Fits, up to 10.0 GB is Tight, and anything more Won't fit.` (R-010 and R-011 examples). |
| How it works | Classification: three bullets, each a rule | three tiles, each a verdict pill, a plain-language line and the same rule | Classification section. Fits: `It fits with the whole reserve to spare.` Tight: `It can run, but it uses part of the reserve, so it is close to your VRAM limit.` Won't fit: `It needs more VRAM than you have.` The rules are unchanged. |
| How it works | (no worked example) | Worked example: Llama 3.1 8B Instruct at Q4, with a 2048-token context. See its model page.  Inputs params_b 8, bits_effective 4.5, kv_gb_per_1k_ctx 0.48, context_tokens 2048. Steps: Weights 4.95 GB (8 x 4.5 / 8 x 1.10 = 4.95); KV cache 0.98 GB (0.48 x 2048 / 1000 = 0.98304); Runtime overhead 1.40 GB (1.0 + 0.05 x 8 = 1.40); Required VRAM 7.33 GB (4.95 + 0.98304 + 1.40 = 7.33304) | Worked example section (R-075). The model name and the link `See its model page` go away when the catalog has no such model (Q-12); the page then says `an 8B model at Q4`. |

### Word for word (R-078), not in the table because nothing changed

- Error messages: `Please fix the following input issues:`, `Invalid value for vram_gb.`, and the other R-008 messages.
- GPU search messages (R-003): `GPU search "RTX" matched multiple entries (...). Select an exact GPU from the list or enter VRAM manually.` and `GPU "RTX 3060 12GB" was not found. Using 12 GB inferred from your input.`
- The default note (R-005): `No GPU or VRAM selected yet. Showing estimates with an assumed 8 GB baseline.`
- Verdict labels `Fits`, `Tight`, `Won't fit`; every figure, margin (`8.6 GB spare`, `0.7 GB spare`, `2.4 GB over`), `15.36 GB needed` and `24.0 GB` captions, and the counts.
- Field and button labels: `Search GPU`, `Clear`, `Available VRAM (GB)`, `Context window`, `2k`, `4k`, `8k`, `RTX 4060 8GB`, `8GB`, `24GB`, `Check compatibility`, `Min params (B)`, `Max params (B)`, `Recommended only`, `Apply filters`, `Family`, `Quantization`, `Recompute`, `GPU`, `Manual VRAM`, `Manual VRAM (GB)`, `Context`, `2048 tokens`.
- Link text and tags: `VRAM Sherpa`, `Estimator`, `How it works`, `Dark mode`, `Light mode`, `Gated` (card) and `gated` (detail), `Active filters:`, the chips (`Family: qwen`, `Quant: Q4`, `Min params >= 8B`, `Max params <= 8B`), `Try Q4`, `Lower context`, `Top Picks for you`, the pick titles, `<n>B · Q4`, `How this is estimated`, `Open to load the breakdown.`.
- Empty and unseeded states: `No models currently fit this setup.`, `Try one of these next steps:`, `Catalog is not initialized yet.`, `Seed the catalog data, then refresh this page.`
- Model detail: `KV cache estimate per 1k context: 0.480 GB`, `Sources`, `Choosing a GPU and entering VRAM together will prioritize the selected GPU.`, the gated note, the seven column headings, `Q4 (Q4) *`.
- How it works: the heading `How VRAM Sherpa estimates fit (Policy v0.1)`, the intro sentence, the section names Inputs, Formula, Reserve buffer, Classification, Limitations, the formulas, and the limitations sentence.
- Footer: `Catalog version: 2026-10-23`, `Estimation policy: v0.1`, `Last updated: 2026-10-23` (and `unseeded` when unseeded).

## Notes for the build

- Tokens: `tokens.css` becomes the first section of `static/css/styles.css` (ADR-0001); `components.css` is the rest. The mockups use only those two.
- Font: `static/fonts/outfit-latin-wght-normal.woff2`; add `static/fonts/*` and `static/img/*` to package-data (ADR-0002, ADR-0003); bump `ASSET_VERSION` (rule 21).
- Logo: `final-mark.svg` inline in `templates/logo_mark.html` (header, `aria-hidden`); `final-favicon.svg` plus a 32 px PNG and a 180 px apple-touch-icon exported from it.
- Theme toggle icon is a CSS mask keyed on `aria-pressed`, so `app.js` can keep replacing the button text.
- Filters: the inputs sit inside the one form, in a `<details>` rendered closed, summary `Filters (N)`. Top Picks: `<details>` closed on phones, opened by an inline script from 900 px.
- The quantisation and context thumbs are progressive (`app.js` adds a `.thumb`). Without JavaScript the pressed control carries its own fill.
- The backdrop is `position: fixed`; the renders stretch the viewport to the page height so it covers the whole image.
