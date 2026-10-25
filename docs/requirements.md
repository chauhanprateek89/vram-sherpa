# Requirements

Each requirement has a number, a status, and EARS criteria that QA can test. Numbers are never reused or renumbered.

| Status | Meaning |
| --- | --- |
| `as-built` | Exists today; the code or tests show it |
| `inferred` | Believed to exist from docs or PRs only. Needs the user's confirmation |
| `planned` | Agreed, not built yet |
| `done` | Built and passed QA |
| `dropped` | Struck by the user; kept so the number is never reused |

EARS forms: `THE SYSTEM SHALL …` (always) · `WHEN <trigger>, THE SYSTEM SHALL …` (event) · `WHILE <state>, THE SYSTEM SHALL …` (state) · `IF <unwanted condition>, THEN THE SYSTEM SHALL …` (error) · `WHERE <feature is present>, THE SYSTEM SHALL …` (option).

Recorded at onboarding from `main` at 96572b1 (2026-10-25). Seeded catalog at that commit: version `2026-10-23`, 35 GPUs, 95 models (81 auto-discovered, 24 gated), 380 variants, 7 families. Paths are relative to the repo root; `main.py` means `src/vramsherpa/main.py`.

R-067 to R-079 (UI/UX overhaul, Full track) were added on 2026-10-25 from the user's intake answers. The user's Gate 1 answers to their open questions (Q-1 to Q-11) are folded into them, and R-080 was added from Q-9; see "Resolved at Gate 1" at the end of this file. At Gate 2 round 1 the R-075 worked example was changed to a catalog model and R-081 was added. The round 1 redo answers (direction "D layout + E glass", Q-12, Q-13 and the old-link redirect) are folded into R-067, R-068, R-075, R-081 and R-032; see "Open questions" and `docs/decisions.md` (rows dated 2026-10-25, Gate "2 (round 1 redo)"). The Gate 2 approval (the 24 approved screens and states, the copy table, filters closed at both widths, the 900 px allowance for message states and the styled GPU suggestion list) is folded into R-067, R-068, R-069, R-071 and R-078; see `docs/decisions.md` (rows dated 2026-10-25, Gate "2").

## Hardware and context input

### R-001 GPU search field

Status: as-built · Track: onboard · UI: yes · Source: `src/vramsherpa/templates/index.html`, `src/vramsherpa/static/js/app.js` (`initGpuInputs`), `main.py` (`_load_gpus`, `_gpu_option_label`)

- THE SYSTEM SHALL show a "Search GPU" text field whose suggestion list holds every catalog GPU, ordered by VRAM ascending, then by name.
- THE SYSTEM SHALL label each suggestion as `<vendor> <name> (<vram to 1 decimal> GB)`, for example `NVIDIA GeForce RTX 4090 (24.0 GB)`.
- WHEN the user's text exactly equals a suggestion, THE SYSTEM SHALL set that GPU as the selected GPU and empty the "Available VRAM (GB)" field.
- WHEN the user's text stops matching a suggestion exactly, THE SYSTEM SHALL clear the selected GPU.
- WHEN the user presses "Clear" beside the field, THE SYSTEM SHALL empty the field, clear the selected GPU and move focus to the field.

### R-002 Manual VRAM field

Status: as-built · Track: onboard · UI: yes · Source: `index.html`, `app.js` (`initGpuInputs`)

- THE SYSTEM SHALL show an "Available VRAM (GB)" number field with minimum 1 and step 0.1.
- WHEN the user types a value into "Available VRAM (GB)", THE SYSTEM SHALL clear the selected GPU.
- WHEN the user changes the GPU search text or the VRAM field, THE SYSTEM SHALL NOT recompute results until the user submits the form with "Check compatibility".

### R-003 Free-text GPU search resolution

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_resolve_available_vram`, `_extract_vram_hint`); tests `test_results_gpu_search_does_not_auto_pick_ambiguous_match`, `test_results_gpu_search_shows_not_found_feedback`, `test_results_gpu_search_uses_vram_hint_from_text`

- WHEN `/results` receives `gpu_search` text and no `gpu_id` or `vram_gb`, THE SYSTEM SHALL compare it case-insensitively, with whitespace collapsed, against each GPU's full label, `<vendor> <name>` and `<name>`.
- WHEN exactly one GPU matches one of those forms exactly, THE SYSTEM SHALL use that GPU's VRAM and mark it as the selected GPU.
- WHEN no exact match exists and exactly one GPU contains the text, THE SYSTEM SHALL use that GPU's VRAM and mark it as the selected GPU.
- WHEN more than one GPU contains the text (for example `RTX`), THE SYSTEM SHALL use 8 GB, leave the selected GPU empty, and show `GPU search "<text>" matched multiple entries (<up to 3 labels>). Select an exact GPU from the list or enter VRAM manually.`
- WHEN no GPU contains the text and the text includes a positive number followed by `GB` (for example `RTX 3060 12GB`), THE SYSTEM SHALL use that number as available VRAM and show `GPU "<text>" was not found. Using 12 GB inferred from your input.`
- WHEN no GPU contains the text and it carries no GB value, THE SYSTEM SHALL use 8 GB and show `GPU "<text>" was not found. Select a listed GPU or enter VRAM manually.`

### R-004 Hardware precedence on the results page

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_resolve_available_vram`, `_selected_hardware_label`)

- WHEN `/results` receives a valid `vram_gb`, THE SYSTEM SHALL use it as available VRAM even if `gpu_id` or `gpu_search` is also present.
- WHEN `/results` receives no valid `vram_gb` and a `gpu_id` that exists in the catalog, THE SYSTEM SHALL use that GPU's VRAM.
- WHEN `/results` receives none of a valid `vram_gb`, a known `gpu_id` or `gpu_search` text, THE SYSTEM SHALL use 8 GB.
- THE SYSTEM SHALL label the hardware in the summary as the selected GPU's label, else `Manual VRAM (<n> GB)`, else `Assumed VRAM (<n> GB)`.

### R-005 Default 8 GB assumption

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`home`, `_results_context`), `results_summary.html`; test `test_home_returns_200`

- WHEN a user opens `/` or `/results` with no GPU, no VRAM and no GPU search text, THE SYSTEM SHALL compute results for 8 GB at 2048 tokens.
- WHILE that default is in use, THE SYSTEM SHALL show `No GPU or VRAM selected yet. Showing estimates with an assumed 8 GB baseline.`
- WHEN a GPU, a VRAM value or GPU search text is supplied, THE SYSTEM SHALL NOT show that sentence.

### R-006 Context window presets

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`CONTEXT_OPTIONS`, `_context_label`), `index.html`

- THE SYSTEM SHALL offer exactly three context presets, 2048, 4096 and 8192 tokens, shown as `2k`, `4k` and `8k`.
- THE SYSTEM SHALL select 2048 tokens by default.
- IF `context_tokens` is an integer other than 2048, 4096 or 8192, THEN THE SYSTEM SHALL use 2048.

### R-007 Example hardware chips

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_example_chips`), `app.js` (`initExampleChips`)

- THE SYSTEM SHALL show three example chips below the inputs: a GPU chip, `8GB` and `24GB`.
- WHERE the catalog holds a GPU whose name contains `4060` with at most 8.1 GB, THE SYSTEM SHALL label the GPU chip `RTX 4060 8GB` and bind it to that GPU; otherwise it SHALL label it `Example GPU` and bind it to the first GPU in VRAM order.
- WHEN the user presses the GPU chip, THE SYSTEM SHALL fill the GPU search with that GPU's label, select it, and empty the VRAM field.
- WHEN the user presses `8GB` or `24GB`, THE SYSTEM SHALL put 8 or 24 in the VRAM field and clear the selected GPU.

### R-008 Input validation on results

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_results_page_context`, `_parse_optional_float`); tests `test_results_accepts_blank_numeric_query_values`, `test_results_invalid_values_render_inline_errors`, `test_results_invalid_range_returns_empty_state`

- WHEN `vram_gb`, `min_params_b` or `max_params_b` is blank, THE SYSTEM SHALL treat it as not supplied.
- IF `vram_gb` is not a number, THEN THE SYSTEM SHALL return HTTP 200 with the inline error `Invalid value for vram_gb.` and ignore the value.
- IF `vram_gb` is below 0.000001, THEN THE SYSTEM SHALL return HTTP 200 with an inline error that `vram_gb` must be greater than or equal to the minimum, and ignore the value.
- IF `min_params_b` or `max_params_b` is not a number or is below 0, THEN THE SYSTEM SHALL show an inline error naming the field and ignore that value.
- IF `min_params_b` is greater than `max_params_b`, THEN THE SYSTEM SHALL show `min_params_b must be less than or equal to max_params_b.` and apply both bounds, which yields no results.
- THE SYSTEM SHALL list inline errors under `Please fix the following input issues:` in an element with `role="alert"`.
- IF `context_tokens` is not an integer or `recommended_only` is not a boolean, THEN THE SYSTEM SHALL return HTTP 422.

## Estimation policy

### R-009 Required VRAM formula

Status: as-built · Track: onboard · UI: no · Source: `src/vramsherpa/estimation.py`; test `test_estimate_required_vram_formula`

- THE SYSTEM SHALL compute `weights_gb = params_b × bits_effective ÷ 8 × 1.10`.
- THE SYSTEM SHALL compute `kv_cache_gb = kv_gb_per_1k_ctx × context_tokens ÷ 1000`.
- THE SYSTEM SHALL compute `runtime_overhead_gb = 1.0 + 0.05 × params_b`.
- THE SYSTEM SHALL compute `required_vram_gb = weights_gb + kv_cache_gb + runtime_overhead_gb`.
- WHEN params_b = 8, bits_effective = 4.5, kv_gb_per_1k_ctx = 0.18 and context = 2048, THE SYSTEM SHALL compute required_vram_gb = 6.71864.

### R-010 Reserve buffer

Status: as-built · Track: onboard · UI: no · Source: `estimation.py` (`reserve_gb`); test `test_reserve_floor_and_percent`

- THE SYSTEM SHALL compute `reserve_gb = max(0.75, 0.10 × available_vram_gb)`.
- WHEN available VRAM is 4 GB, THE SYSTEM SHALL use a 0.75 GB reserve; WHEN it is 16 GB, a 1.6 GB reserve.

### R-011 Fit classification

Status: as-built · Track: onboard · UI: yes · Source: `estimation.py` (`classify_fit`); test `test_classification_boundaries`

- WHEN `required_vram_gb ≤ available_vram_gb − reserve_gb`, THE SYSTEM SHALL classify the variant `Fits`.
- WHEN `available_vram_gb − reserve_gb < required_vram_gb ≤ available_vram_gb`, THE SYSTEM SHALL classify it `Tight`.
- WHEN `required_vram_gb > available_vram_gb`, THE SYSTEM SHALL classify it `Won't fit`.
- WHEN available VRAM is 10 GB, THE SYSTEM SHALL classify 9.0 GB as `Fits`, 9.0001 GB and 10.0 GB as `Tight`, and 10.0001 GB as `Won't fit`.

### R-012 Margin

Status: as-built · Track: onboard · UI: yes · Source: `estimation.py` (`estimate_breakdown`)

- THE SYSTEM SHALL compute `margin_gb = available_vram_gb − required_vram_gb`, negative when the variant exceeds available VRAM.

## Results

### R-013 Results on the home page and at /results

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`home`, `results`); tests `test_home_returns_200`, `test_results_with_manual_vram_returns_200_and_rows`

- WHEN a user opens `/`, THE SYSTEM SHALL render the input form and results for the default 8 GB, 2048 tokens and no filters, ignoring any query parameters.
- WHEN a user opens `/results` with query parameters, THE SYSTEM SHALL render the same page with results for those parameters.
- WHEN the form is submitted without JavaScript, THE SYSTEM SHALL send a GET to `/results` carrying every field as a query parameter.

### R-014 One card per model, usable models only

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_group_results_by_model`), `results_list.html`; PR #19

- THE SYSTEM SHALL show one card per model, not one per variant.
- THE SYSTEM SHALL show a model's card only when at least one of its variants (after filters) is `Fits` or `Tight`.
- THE SYSTEM SHALL state above the cards `<N> model(s) fit or are tight. Pick a quantisation on any card to see how it changes.`, using `model` when N is 1.

### R-015 Card contents

Status: as-built · Track: onboard · UI: yes · Source: `results_list.html`, `main.py` (`ModelCard`, `_results_context`)

- THE SYSTEM SHALL show on each card the model name, the parameter count as `<n>B params` with no trailing zeros, and the family.
- THE SYSTEM SHALL link the model name to `/models/<model_id>` with the current context and hardware (`gpu_id`, `gpu_search` or `vram_gb`) in the query string.
- THE SYSTEM SHALL show one quantisation button per variant on the card, ordered Q4, Q5, Q8, FP16.

### R-016 Card default quantisation

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_group_results_by_model`)

- WHEN the model's recommended variant is `Fits` or `Tight`, THE SYSTEM SHALL open the card on that variant.
- WHEN the recommended variant is absent or `Won't fit`, THE SYSTEM SHALL open the card on the first usable variant in Q4, Q5, Q8, FP16 order.
- THE SYSTEM SHALL mark the open variant's button with `aria-pressed="true"` and the others with `aria-pressed="false"`.

### R-017 Quantisation switcher

Status: as-built · Track: onboard · UI: yes · Source: `app.js` (quant click handler), `results_list.html`, `styles.css` (`[data-oversize]`)

- WHEN the user presses a quantisation button, THE SYSTEM SHALL show that variant's fit pane, hide the card's other panes and update `aria-pressed`, without a network request.
- THE SYSTEM SHALL strike through the button of each `Won't fit` variant and give it the tooltip `Does not fit in <available, 0 decimals> GB`.

### R-018 Fit pane and gauge

Status: as-built · Track: onboard · UI: yes · Source: `results_list.html`, `main.py` (`ResultRow` gauge properties)

- THE SYSTEM SHALL show the variant's classification (`Fits`, `Tight` or `Won't fit`).
- WHEN margin ≥ 0, THE SYSTEM SHALL show `<margin, 1 decimal> GB spare`; WHEN margin < 0, `<−margin, 1 decimal> GB over`.
- THE SYSTEM SHALL draw a bar whose filled width is `min(100, required ÷ available × 100)` percent, with a reserve band covering the last `reserve ÷ available` of the track.
- THE SYSTEM SHALL caption the gauge `<required, 2 decimals> GB needed` and `<available, 1 decimal> GB`.
- THE SYSTEM SHALL give the gauge `role="img"` and the label `Needs <required> of <available> gigabytes, with <reserve> gigabytes held in reserve`.
- THE SYSTEM SHALL render the gauge from server-computed inline styles, so it displays without JavaScript.

### R-019 Card ordering

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_group_results_by_model`)

- THE SYSTEM SHALL order cards by the open variant's classification (`Fits` before `Tight`), then by parameter count descending, then by model name ascending.

### R-020 "How this is estimated" breakdown

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`variant_breakdown`), `results_why.html`, `results_list.html`; test `test_variant_breakdown_returns_html_partial`

- THE SYSTEM SHALL show a collapsed `How this is estimated` disclosure in each fit pane.
- WHEN the user first opens it, THE SYSTEM SHALL fetch `/results/why/<variant_id>` once and show `<quant label> on <model name>` with weights_gb, kv_cache_gb, runtime_overhead_gb, reserve_gb and required_vram_gb, each to 2 decimals.
- THE SYSTEM SHALL compute the breakdown from the `gpu_id`, `vram_gb` and `context_tokens` in that request, with the same precedence and context fallback as R-004 and R-006.
- IF the variant id is unknown, THEN THE SYSTEM SHALL return HTTP 404 `{"detail": "Variant not found"}`.
- IF `vram_gb` is invalid on that request, THEN THE SYSTEM SHALL return HTTP 422.

### R-021 Gated models flagged on cards

Status: as-built · Track: onboard · UI: yes · Source: `results_list.html`, `src/vramsherpa/models.py` (`Model.gated`); PR #14

- WHEN a model is gated, THE SYSTEM SHALL show a `Gated` tag beside its name with the tooltip `Accept the model licence on Hugging Face before downloading the weights.`
- THE SYSTEM SHALL list gated models with the same estimates as ungated ones, not exclude them.

## Summary and guidance

### R-022 Summary banner

Status: as-built · Track: onboard · UI: yes · Source: `results_summary.html`, `main.py` (`_results_context`)

- THE SYSTEM SHALL show `Summary for this query` with `Hardware: <label>` (R-004) and `Context: <2k|4k|8k>`.
- THE SYSTEM SHALL state `Showing only Fits and Tight models in the list below.`
- WHEN GPU search produced feedback (R-003), THE SYSTEM SHALL show it in an element with `role="status"`.

### R-023 Summary counts per model

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_summary_counts`); PR #19

- THE SYSTEM SHALL show three counts, `Fits`, `Tight` and `Won't fit`, counting models, not variants.
- THE SYSTEM SHALL count each model under the classification of the variant its card opens on (R-016).
- THE SYSTEM SHALL count a model with no usable variant under `Won't fit`.
- THE SYSTEM SHALL make the `Fits` and `Tight` counts sum to the number of cards shown.

### R-024 Top picks

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_top_picks`, `_build_results`), `results_summary.html`

- THE SYSTEM SHALL rank usable variants by: recommended first, then `Fits` before `Tight`, then parameter count descending, then bits_effective descending, then model name.
- WHEN at least one variant is `Fits` and at least three variants are `Fits` or `Tight`, THE SYSTEM SHALL show `Top Picks for you` with three cards: `Best quality that fits` (the top-ranked `Fits` variant), `Best balance` (the top-ranked usable variant other than the first) and `Best small` (the lowest-ranked usable variant not already picked).
- WHEN fewer than three usable variants exist or none is `Fits`, THE SYSTEM SHALL hide the Top Picks section.
- THE SYSTEM SHALL show on each pick the model name linked to its detail page, `<params, 1 decimal>B · <quant label>`, and a one-line description.

### R-025 Empty state

Status: as-built · Track: onboard · UI: yes · Source: `results_list.html`, `main.py` (`_empty_state_suggestions`); test `test_results_invalid_range_returns_empty_state`

- WHEN the catalog is seeded and no card is shown, THE SYSTEM SHALL show `No models currently fit this setup.` and `Try one of these next steps:`.
- WHEN the quantisation filter is anything other than exactly `Q4`, THE SYSTEM SHALL make `Try Q4` a link to the same query with the quantisation filter set to `Q4` only; otherwise it SHALL show `Try Q4` as inert text.
- WHEN a smaller context preset exists, THE SYSTEM SHALL make `Lower context` a link to the same query at the next smaller preset; at 2048 tokens it SHALL show `Lower context` as inert text.
- WHEN the `Won't fit` count is above 0, THE SYSTEM SHALL add `There are <n> variants in Won't fit, hidden by default.`

### R-026 Unseeded catalog state

Status: as-built · Track: onboard · UI: yes · Source: `results_list.html`, `main.py` (`_catalog_version`); test `test_home_unseeded_shows_catalog_not_initialized`

- WHILE the catalog has no `catalog_version` record, THE SYSTEM SHALL show `Catalog is not initialized yet.` and `Seed the catalog data, then refresh this page.` in place of results, and SHALL NOT show `No models currently fit this setup.`

## Filters

### R-027 Family filter

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_variant_query`, `_all_families`), `index.html`; test `test_results_filters_change_output_deterministically`

- THE SYSTEM SHALL list one checkbox per distinct family in the catalog, sorted alphabetically.
- WHEN one or more families are checked, THE SYSTEM SHALL show only models in any of the checked families.
- WHEN `family=llama` and 24 GB are requested, THE SYSTEM SHALL list `Llama 3.1 8B Instruct` and not `Qwen2.5 14B Instruct`; WHEN `family=qwen`, the reverse.

### R-028 Quantisation filter

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_variant_query`, `_all_quant_buckets`), `index.html`

- THE SYSTEM SHALL list one checkbox per distinct quantisation bucket in the catalog, sorted alphabetically (`FP16`, `Q4`, `Q5`, `Q8`).
- WHEN one or more buckets are checked, THE SYSTEM SHALL include only variants in the checked buckets, both on cards and in counts.

### R-029 Parameter-size bounds

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_variant_query`), `index.html`

- THE SYSTEM SHALL offer `Min params (B)` and `Max params (B)` number fields, minimum 0, step 0.1.
- WHEN `min_params_b` is set, THE SYSTEM SHALL include only models with params_b ≥ that value; WHEN `max_params_b` is set, only models with params_b ≤ that value.

### R-030 Recommended-only filter

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_variant_query`), `index.html`, `tools/catalog_sources.yaml` (`recommended_bucket: Q4`)

- WHEN `Recommended only` is checked, THE SYSTEM SHALL include only variants flagged recommended, which in the shipped catalog is the Q4 variant of every model.

### R-031 Active filter chips

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`_active_filter_chips`), `results_summary.html`

- WHEN any filter is active, THE SYSTEM SHALL show `Active filters:` followed by one chip per active value: `Family: <f>`, `Quant: <q>`, `Min params >= <n>B`, `Max params <= <n>B`, `Recommended only`.
- WHEN the user follows a chip, THE SYSTEM SHALL load `/results` with that one value removed and every other filter, the hardware and the context unchanged.

### R-032 Live filtering and shareable URLs

Status: as-built · Track: onboard · UI: yes · Source: `index.html` (`hx-get`, `hx-trigger`, `hx-push-url`), `main.py` (`results`, `results_partial_*`); test `test_results_partials_return_summary_and_results_sections`

- WHEN JavaScript is available and the user changes a context preset, a family or quantisation checkbox, or `Recommended only`, THE SYSTEM SHALL refresh the summary and results in place without a full page load.
- WHEN the user types in `Min params (B)` or `Max params (B)`, THE SYSTEM SHALL refresh results 350 ms after the last keystroke.
- WHEN results refresh in place, THE SYSTEM SHALL update the browser address to `/results/partials/content?<query>`, the URL of the fragment request. Correction (Gate 2 round 1 redo, ADR-0008): onboarding recorded this as "a URL that reproduces the same results when opened directly", which the code does not do. Opening or reloading that address today shows the bare results fragment, without the header, the form, the footer or the stylesheet. R-081 plans the fix: the address becomes `/results?<query>`, and old addresses redirect there.
- WHEN a request to `/results` carries `HX-Request: true`, THE SYSTEM SHALL return only the results fragment.
- THE SYSTEM SHALL serve fragments at `/results/partials/content`, `/results/partials/summary` and `/results/partials/list`, accepting the same parameters as `/results`. Today this holds with or without `HX-Request: true`; R-081 plans a redirect for `/results/partials/content` when it is opened as a normal page load.
- WHILE JavaScript is disabled, THE SYSTEM SHALL show an `Apply filters` button that submits the form.

## Model detail

### R-033 Model detail page

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`model_detail`, `_to_source_items`), `model_detail.html`; test `test_model_detail_returns_200`

- WHEN a user opens `/models/<model_id>`, THE SYSTEM SHALL show the model name, family, params to 1 decimal, and type and licence when present.
- THE SYSTEM SHALL show `KV cache estimate per 1k context: <value, 3 decimals> GB`.
- WHERE the model has sources, THE SYSTEM SHALL list them once each, rendering `http://` and `https://` entries as links that open in a new tab with `rel="noopener noreferrer"` and other entries as plain text.
- IF the model id is unknown, THEN THE SYSTEM SHALL return HTTP 404 `{"detail": "Model not found"}`.

### R-034 Variant table on model detail

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`model_detail`), `model_detail.html`

- THE SYSTEM SHALL list every variant of the model, including `Won't fit` ones, sorted by bucket then label as text (`FP16`, `Q4`, `Q5`, `Q8`).
- THE SYSTEM SHALL show per variant: `<label> (<bucket>)` with ` *` appended when recommended, bits to 1 decimal, required, available and margin to 2 decimals, the classification badge, and provenance (notes and sources).

### R-035 Recompute form on model detail

Status: as-built · Track: onboard · UI: yes · Source: `main.py` (`model_detail`), `model_detail.html`, `app.js` (`initModelDetailInputs`); tests `test_model_detail_prefers_gpu_selection_when_both_gpu_and_manual_vram_provided`, `test_model_detail_accepts_blank_manual_vram`, `test_model_detail_invalid_manual_vram_renders_inline_error`

- THE SYSTEM SHALL offer a GPU select (first option `Manual VRAM`), a `Manual VRAM (GB)` field, a context select (`2048 tokens`, `4096 tokens`, `8192 tokens`) and a `Recompute` button.
- WHEN both a known `gpu_id` and `vram_gb` are supplied, THE SYSTEM SHALL use the GPU's VRAM and say so in the help text `Choosing a GPU and entering VRAM together will prioritize the selected GPU.`
- WHEN the user picks a GPU, THE SYSTEM SHALL empty the VRAM field; WHEN the user types VRAM, it SHALL reset the GPU select to `Manual VRAM`.
- WHEN neither is supplied, THE SYSTEM SHALL compute for 8 GB.
- IF `vram_gb` is invalid, THEN THE SYSTEM SHALL return HTTP 200 with the inline error `Invalid value for vram_gb.`

### R-036 Gated notice on model detail

Status: as-built · Track: onboard · UI: yes · Source: `model_detail.html`; tests `test_model_detail_flags_gated_model`, `test_model_detail_omits_gated_notice_for_open_model`

- WHEN the model is gated, THE SYSTEM SHALL show a `gated` badge and `Weights are gated on Hugging Face: accept the model license on its model page before downloading. VRAM estimates below are unaffected.`
- WHEN the model is not gated, THE SYSTEM SHALL show neither.

## How it works

### R-037 How-it-works page

Status: as-built · Track: onboard · UI: yes · Source: `how_it_works.html`; test `test_how_it_works_returns_200`

- WHEN a user opens `/how-it-works`, THE SYSTEM SHALL show the heading `How VRAM Sherpa estimates fit (Policy v0.1)`, the inputs, the four formulas of R-009, the reserve formula of R-010 and the three classification rules of R-011.
- THE SYSTEM SHALL state `Real outcomes vary by runtime, driver stack, memory fragmentation, and CPU/GPU offload behavior. This tool is not a benchmark and does not guarantee performance.`

## Site-wide

### R-038 Trust footer

Status: as-built · Track: onboard · UI: yes · Source: `base.html`, `main.py` (`_base_context`); test `test_home_returns_200`; `README.md` "Trust layer"

- THE SYSTEM SHALL show on every page `Catalog version: <v>`, `Estimation policy: v0.1` and `Last updated: <v>`, where `<v>` is the seeded catalog version.
- WHILE the catalog is unseeded, THE SYSTEM SHALL show `unseeded` as `<v>`.

### R-039 Navigation

Status: as-built · Track: onboard · UI: yes · Source: `base.html`

- THE SYSTEM SHALL show `VRAM Sherpa` (linking to `/`) and the links `Estimator` and `How it works` on every page.
- WHEN the path is `/`, `/results` or starts with `/models/`, THE SYSTEM SHALL mark `Estimator` with `aria-current="page"`; WHEN it is `/how-it-works`, it SHALL mark `How it works`.

### R-040 Colour theme

Status: as-built · Track: onboard · UI: yes · Source: `base.html`, `app.js` (`initThemeToggle`), `styles.css` (`prefers-color-scheme`)

- WHEN a visitor has no saved theme, THE SYSTEM SHALL follow the operating system's light or dark preference.
- WHEN the visitor presses the theme button, THE SYSTEM SHALL switch between light and dark and save the choice in `localStorage` under `vramsherpa-theme`.
- WHEN a saved theme exists, THE SYSTEM SHALL apply it before the page paints.
- THE SYSTEM SHALL label the button `Light mode` while dark and `Dark mode` while light, with `aria-pressed="true"` while dark.

### R-041 Versioned static assets

Status: as-built · Track: onboard · UI: no · Source: `main.py` (`ASSET_VERSION`), `base.html`

- THE SYSTEM SHALL request its CSS and JS with a `?v=<asset version>` query (currently `20261024a`), so a release that changes the version makes browsers fetch the new files.

## Security and configuration

### R-042 Read-only public site

Status: as-built · Track: onboard · UI: no · Source: `main.py` (`create_app`)

- THE SYSTEM SHALL register only GET routes: `/`, `/results`, `/results/partials/content`, `/results/partials/summary`, `/results/partials/list`, `/results/why/{variant_id}`, `/models/{model_id}`, `/how-it-works`, `/healthz` and `/static`.
- THE SYSTEM SHALL offer no login, account or write action to visitors.

### R-043 Host allowlist

Status: as-built · Track: onboard · UI: no · Source: `main.py` (`host_guard`), `src/vramsherpa/config.py`; tests `test_parse_allowed_hosts_defaults_to_wildcard_when_unset`, `test_parse_allowed_hosts_defaults_to_wildcard_when_blank`; `release.yml` smoke test

- WHEN `ALLOWED_HOSTS` is unset, blank or contains only separators, THE SYSTEM SHALL accept any `Host`.
- WHEN `ALLOWED_HOSTS` is a comma-separated list, THE SYSTEM SHALL compare the request's `Host` header, with any port removed, against the list.
- IF the `Host` header is present and not in the list, THEN THE SYSTEM SHALL return HTTP 400 `{"detail": "Host not allowed"}` for every path, including `/healthz` and `/static`.
- THE SYSTEM SHALL decide host acceptance from `ALLOWED_HOSTS` alone; `APP_ENV` and `PUBLIC_HOSTNAME` do not change it.

### R-044 Database URL required

Status: as-built · Track: onboard · UI: no · Source: `main.py` (`create_app`), `src/vramsherpa/seed.py`; tests `test_create_app_requires_database_url_*`

- IF no database URL is configured, THEN THE SYSTEM SHALL refuse to start with `DATABASE_URL must be set when APP_ENV is not 'test'.` (or the `APP_ENV is 'test'` variant).
- IF `DATABASE_URL` is unset when the seed command runs, THEN THE SYSTEM SHALL stop with `DATABASE_URL must be set to run the seed command.`

### R-045 HTTPS asset URLs behind the tunnel

Status: as-built · Track: onboard · UI: no · Source: `docker-compose.prod.yml` (`--proxy-headers --forwarded-allow-ips=*`), `release.yml` smoke test; `docs/RELEASE_AND_DEPLOY.md`

- WHEN a production request carries `X-Forwarded-Proto: https`, THE SYSTEM SHALL render CSS and JS URLs with `https://`, never `http://`.

### R-046 Production container hardening

Status: as-built · Track: onboard · UI: no · Source: `docker-compose.prod.yml`

- THE SYSTEM SHALL publish the web port only on `127.0.0.1:8000` in production.
- THE SYSTEM SHALL NOT publish the database port in production.
- THE SYSTEM SHALL run the production web container with `no-new-privileges` and all Linux capabilities dropped.
- WHERE `POSTGRES_PASSWORD` is unset in production, THE SYSTEM SHALL fall back to the password `vramsherpa`.

## Health

### R-047 Readiness endpoint

Status: as-built · Track: onboard · UI: no · Source: `main.py` (`healthz`); tests `test_healthz_returns_ready_when_seeded`, `test_healthz_returns_503_when_catalog_unseeded`; `README.md`

- WHEN the database is reachable and the catalog has a version record and at least one GPU, model and variant, THE SYSTEM SHALL return HTTP 200 with `status: "ok"`, `ready: true`, `detail: "ready"`, `catalog_version` and `counts` for gpus, models and variants.
- WHEN the database is reachable but the catalog is not seeded, THE SYSTEM SHALL return HTTP 503 with `status: "degraded"`, `ready: false`, `detail: "Catalog is not seeded"`.
- IF the database query fails, THEN THE SYSTEM SHALL return HTTP 503 with `status: "error"`, `ready: false`, `detail: "Database unavailable"` and the exception text in `error`.
- THE SYSTEM SHALL run a container health check against `/healthz` every 10 s, timing out at 5 s, with 10 retries, in both compose files.

## Catalog data and seeding

### R-048 Seed loading

Status: as-built · Track: onboard · UI: no · Source: `src/vramsherpa/catalog.py` (`seed_catalog`); tests `test_seed_loads_and_is_idempotent`, `test_seed_normalizes_list_and_string_text_fields`, `test_seed_defaults_gated_to_false_when_absent`

- THE SYSTEM SHALL load GPUs, models and variants from `seed_gpus.json`, `seed_models.json` and `seed_variants.json` in `SEED_DATA_DIR` (default `data`).
- WHEN seeding runs twice on the same files, THE SYSTEM SHALL leave the GPU, model and variant counts and the catalog version unchanged.
- THE SYSTEM SHALL insert items whose id is new and overwrite every field of items whose id exists.
- THE SYSTEM SHALL NOT delete rows whose id no longer appears in the seed files.
- THE SYSTEM SHALL store list-valued `notes` and `sources` as JSON text and default a missing `gated` to false.
- THE SYSTEM SHALL record the seed files' `catalog_version` as the catalog version.
- IF a seed file is missing or lacks a string `catalog_version` and a list `items`, THEN THE SYSTEM SHALL stop seeding with an error naming the file.
- IF the three files carry different `catalog_version` values, THEN THE SYSTEM SHALL stop seeding with `Catalog versions do not match across seed files`.
- IF a variant names a `model_id` that is not loaded, THEN THE SYSTEM SHALL stop seeding with `Variant references missing model_id=<id>`.

### R-049 Seed on start and seed command

Status: as-built · Track: onboard · UI: no · Source: `seed.py`, `docker-compose.yml`, `docker-compose.prod.yml`, `pyproject.toml` (`vramsherpa-seed`)

- WHEN the web container starts, THE SYSTEM SHALL create missing tables, run the seed, and start the web server only if seeding succeeded.
- WHEN seeding succeeds, THE SYSTEM SHALL print `Seeded catalog_version=<v> with <g> GPUs, <m> models, <n> variants`.
- THE SYSTEM SHALL expose the seed as `python -m vramsherpa.seed` and as the `vramsherpa-seed` command.

### R-050 Catalog ships inside the image

Status: as-built · Track: onboard · UI: no · Source: `Dockerfile` (`COPY data ./data`); `docs/RELEASE_AND_DEPLOY.md` "Catalog refreshes"

- THE SYSTEM SHALL bake `data/seed_*.json` into each image, so running an older release tag seeds the catalog that shipped with it.

### R-051 Catalog minimum contents

Status: as-built · Track: onboard · UI: no · Source: test `test_seed_loads_and_is_idempotent`, `test_seed_round_trips_gated_flag`; `data/seed_*.json`

- THE SYSTEM SHALL ship at least 20 GPUs, at least 3 model families and at least 50 variants.
- THE SYSTEM SHALL ship at least one gated and one ungated model; `model_llama_31_8b_instruct` SHALL be gated and `model_qwen25_7b_instruct` ungated.
- THE SYSTEM SHALL ship exactly four variants per model, Q4, Q5, Q8 and FP16, with the Q4 variant recommended.

### R-052 Schema upgrades on an existing database

Status: as-built · Track: onboard · UI: no · Source: `README.md` "Upgrading an existing database", PR #14

- WHEN a release adds a column and the database volume predates it, THE SYSTEM SHALL fail seeding (Postgres `UndefinedColumn`) rather than alter the table.
- WHEN the operator recreates the volume or adds the column by hand with `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, THE SYSTEM SHALL seed normally on the next start.

## Catalog refresh tooling

### R-053 Seed generation from curated YAML

Status: as-built · Track: onboard · UI: no · Source: `tools/ingest_catalog.py` (`_build_gpus`, `_build_models`, `_build_variants`, `_normalize_quant_bits`), `tools/catalog_sources.yaml`

- WHEN `tools/ingest_catalog.py --config tools/catalog_sources.yaml --output-dir data` runs, THE SYSTEM SHALL write `seed_gpus.json`, `seed_models.json` and `seed_variants.json`, each as `{"catalog_version": "YYYY-MM-DD", "items": [...]}`.
- THE SYSTEM SHALL generate, for every model, one variant per bucket Q4, Q5, Q8 and FP16, with bits_effective 4.5, 5.5, 8.5 and 16.0, quant label equal to the bucket, and id `variant_<model id>_<bucket in lower case>`.
- THE SYSTEM SHALL flag the configured `recommended_bucket` (default `Q4`) as recommended on each model.
- IF the configured buckets or bits differ from exactly Q4 = 4.5, Q5 = 5.5, Q8 = 8.5, FP16 = 16.0, or the recommended bucket is not one of them, THEN THE SYSTEM SHALL stop with an error naming the mismatch.
- IF a curated GPU or model lacks a required string or number field, THEN THE SYSTEM SHALL stop with an error naming the entry and field.
- WHERE a curated entry has no `id`, THE SYSTEM SHALL derive `gpu_<vendor>_<name>_<int vram>gb` or `model_<name>` from slugified fields.
- WHEN generation finishes, THE SYSTEM SHALL validate the written files (R-058) and print `Catalog refresh completed for <v>: <g> GPUs, <m> models, <n> variants`.

### R-054 Hugging Face model discovery

Status: as-built · Track: onboard · UI: no · Source: `ingest_catalog.py` (`_discover_models_from_hf`, `_extract_params_b`, `_fetch_hf_model_list`), `catalog_sources.yaml` (`discovery`); `tests/test_ingest_catalog.py`; PRs #10, #13, #14

- WHERE `discovery.enabled` is true, THE SYSTEM SHALL fetch, for each tracked organisation (meta-llama, mistralai, Qwen, google, microsoft, 01-ai, deepseek-ai), its top 50 text-generation repos by downloads from the Hugging Face API.
- THE SYSTEM SHALL skip a repo that is already curated or already in the catalog, has fewer than 50,000 downloads, does not contain the organisation's name filter (`gemma` for google, `phi` for microsoft), or contains any of `Guard`, `shield`, `-AWQ`, `-GPTQ`, `-FP8`, `.gguf` (case-insensitive).
- THE SYSTEM SHALL take params_b from `discovery.params_b_overrides` when the repo id or name matches (case-insensitive), else from an `<n>B` token in the repo name between 0.1 and 2000, else skip the repo.
- THE SYSTEM SHALL add at most 20 new models per organisation per run.
- THE SYSTEM SHALL keep gated repos, set `gated: true`, and prefix their notes with `Gated repo: license acceptance required to download weights.`
- THE SYSTEM SHALL set kv_gb_per_1k_ctx to `round(0.06 × params_b, 2)`, model type `instruct` when the name contains `-instruct`, `-chat`, `-it` or `chat-hf` and `base` otherwise, and licence from the repo card, else the organisation default.
- THE SYSTEM SHALL record the repo URL and the KV heuristic in each discovered model's sources and mark it `discovered: true`.
- IF fetching one organisation's list fails after retries, THEN THE SYSTEM SHALL print a warning and continue with the other organisations.
- THE SYSTEM SHALL print `Discovery: added <n> new model(s) from HuggingFace, carried over <m> from the previous catalog.`

### R-055 Discovered models persist across refreshes

Status: as-built · Track: onboard · UI: no · Source: `ingest_catalog.py` (`_load_discovered_models`); tests `test_load_discovered_models_*`; PR #13

- WHEN a refresh runs, THE SYSTEM SHALL carry every previously discovered model from the existing `seed_models.json` into the new catalog.
- WHEN a carried model's id or Hugging Face id is now curated in the YAML, THE SYSTEM SHALL keep the curated entry and drop the carried one.
- THE SYSTEM SHALL list curated models in YAML order followed by discovered models sorted by id.
- WHEN a refresh runs with no upstream change, THE SYSTEM SHALL write byte-identical seed files.

### R-056 Catalog version on refresh

Status: as-built · Track: onboard · UI: no · Source: `ingest_catalog.py` (`_resolve_catalog_version`)

- WHERE `--catalog-version` is given, THE SYSTEM SHALL use it.
- WHEN the generated items equal those in all three existing seed files, THE SYSTEM SHALL keep the existing catalog version.
- WHEN any item differs, THE SYSTEM SHALL set the catalog version to the current UTC date as `YYYY-MM-DD`.

### R-057 Optional Hugging Face metadata enrichment

Status: as-built · Track: onboard · UI: no · Source: `ingest_catalog.py` (`_enrich_models_from_hf`, `_fetch_hf_model_metadata`)

- WHERE `--with-hf-metadata` is passed or `catalog.huggingface.enabled` is true, THE SYSTEM SHALL fetch metadata for each model with a Hugging Face repo, fill an empty licence from the repo card, and add `https://huggingface.co/<repo>` to its sources.
- IF a metadata fetch fails, THEN THE SYSTEM SHALL print a warning and continue, unless `--strict-hf` is passed, in which case it SHALL stop with an error.

### R-058 Catalog validation

Status: as-built · Track: onboard · UI: no · Source: `tools/validate_catalog.py`; `ci.yml` step "Validate catalog seeds"

- WHEN `tools/validate_catalog.py --data-dir data --config tools/catalog_sources.yaml` runs, THE SYSTEM SHALL exit 1 and list every error if any rule below fails, and otherwise print `Catalog validation passed: <g> GPUs, <m> models, <n> variants` and exit 0.
- THE SYSTEM SHALL require each file's `catalog_version` to be a `YYYY-MM-DD` date and the same in all three files.
- THE SYSTEM SHALL require unique, non-empty ids within each file.
- THE SYSTEM SHALL require GPUs to have non-empty vendor and name, numeric vram_gb and a list of non-empty source strings.
- THE SYSTEM SHALL require models to have non-empty name and family, numeric params_b and kv_gb_per_1k_ctx, `license` and `model_type` keys, a boolean `gated` and a sources list.
- THE SYSTEM SHALL require variants to reference an existing model, use a bucket of Q4, Q5, Q8 or FP16 with the matching bits_effective, have a non-empty quant label, a boolean `recommended` and a sources list.

### R-059 Weekly catalog refresh pull request

Status: as-built · Track: onboard · UI: no · Source: `.github/workflows/catalog-refresh.yml`; `README.md` "Scheduled catalog PR workflow"

- THE SYSTEM SHALL run the catalog refresh every Sunday at 08:00 UTC and on manual dispatch.
- WHERE a manual run sets `include_hf_metadata` to true, THE SYSTEM SHALL pass `--with-hf-metadata`; scheduled runs SHALL NOT.
- THE SYSTEM SHALL run generation and then validation, and stop the run if either fails.
- WHEN any of the three seed files changed, THE SYSTEM SHALL open or update a PR from branch `automation/catalog-refresh` titled `Catalog refresh YYYY-MM-DD`, labelled `catalog` and `automated-pr`, whose body lists the GPU, model and variant counts and the discovery line.
- WHEN no seed file changed, THE SYSTEM SHALL open no PR and log `No catalog changes detected.`

## Continuous integration, release and deploy

### R-060 CI checks

Status: as-built · Track: onboard · UI: no · Source: `.github/workflows/ci.yml`; `tests/conftest.py`

- WHEN a pull request is opened or updated, or `release.yml` calls it, THE SYSTEM SHALL run on Python 3.12: `ruff check .`, `ruff format --check .`, catalog validation and `pytest`, failing on the first that fails.
- THE SYSTEM SHALL run the tests against a temporary SQLite database, independent of any developer or production database.

### R-061 Release versioning and image tags

Status: as-built · Track: onboard · UI: no · Source: `.github/workflows/release.yml`; `docs/RELEASE_AND_DEPLOY.md` "Versioning"

- WHEN a commit lands on `main` or the release is dispatched manually, THE SYSTEM SHALL run the CI checks of R-060 and release only if they pass.
- THE SYSTEM SHALL name the release `v<major>.<minor>.<commit count on main>`, taking major and minor from `version` in `pyproject.toml`.
- IF that tag already exists, THEN THE SYSTEM SHALL fail the release.
- THE SYSTEM SHALL push the image to `ghcr.io/chauhanprateek89/vram-sherpa` with three tags: the version, `sha-<short commit>`, and `latest`.
- THE SYSTEM SHALL run at most one release at a time and SHALL NOT cancel a release in progress.

### R-062 Release smoke test

Status: as-built · Track: onboard · UI: no · Source: `release.yml` step "Smoke test the published image"

- WHEN the image is pushed, THE SYSTEM SHALL start it with `APP_ENV=prod` and `ALLOWED_HOSTS=localhost,127.0.0.1` against a fresh Postgres 16.
- IF `/healthz` does not return success within 30 attempts 2 s apart, THEN THE SYSTEM SHALL fail the release.
- IF the container log lacks `Seeded catalog_version=`, THEN THE SYSTEM SHALL fail the release.
- IF `GET /?vram_gb=24&context_tokens=4096` does not return success, THEN THE SYSTEM SHALL fail the release.
- IF any static asset URL on `/`, requested with `X-Forwarded-Proto: https`, starts with `http://`, THEN THE SYSTEM SHALL fail the release.
- IF a request with `Host: evil.example.com` returns anything other than HTTP 400, THEN THE SYSTEM SHALL fail the release.
- WHEN the smoke test passes, THE SYSTEM SHALL push the git tag and create a GitHub Release with generated notes.
- IF the smoke test fails, THEN THE SYSTEM SHALL create no git tag and no GitHub Release.

### R-063 Pull-based deploy to the homelab

Status: as-built · Track: onboard · UI: no · Source: `docker-compose.prod.yml`; `docs/RELEASE_AND_DEPLOY.md`; PRs #15, #17, #18

- THE SYSTEM SHALL run the production web container from `ghcr.io/chauhanprateek89/vram-sherpa:${IMAGE_TAG:-latest}`, pulling on every start and restarting unless stopped.
- THE SYSTEM SHALL run watchtower `nickfedor/watchtower:1.23.2` with `--label-enable --cleanup --interval 300`, so a new `latest` is running within 5 minutes of being pushed.
- THE SYSTEM SHALL label only the web container for auto-update; the database container SHALL never be auto-updated.
- THE SYSTEM SHALL pin the database image to `postgres:16-alpine` and the watchtower image to `1.23.2`.
- THE SYSTEM SHALL keep the production compose file free of any `build:` step.

### R-064 Rollback to a pinned release

Status: as-built · Track: onboard · UI: no · Source: `docker-compose.prod.yml` (`IMAGE_TAG`); `docs/RELEASE_AND_DEPLOY.md` "Rolling back"

- WHEN `IMAGE_TAG=<version>` is set in the server's `.env` and the web service is recreated, THE SYSTEM SHALL run that release's image and its bundled catalog.
- WHEN `IMAGE_TAG` is removed, THE SYSTEM SHALL return to tracking `latest`.

### R-065 First-time server setup

Status: as-built · Track: onboard · UI: no · Source: `docs/RELEASE_AND_DEPLOY.md` "First-time server setup", PR #15

- THE SYSTEM SHALL be published as a public GHCR package, so the server pulls without registry credentials.
- THE SYSTEM SHALL read its production settings from a `.env` beside `docker-compose.prod.yml` with permissions `600`.
- THE SYSTEM SHALL be reached publicly through a Public Hostname on the existing cloudflared systemd tunnel pointing at `http://localhost:8000`, with no tunnel container in the stack.
- WHEN a rollback is in effect, THE SYSTEM SHALL have watchtower stopped, so it does not move the web container off the pinned tag.

### R-066 Local development stack

Status: as-built · Track: onboard · UI: no · Source: `docker-compose.yml`; `README.md` "Run with Docker Compose"

- WHEN a developer runs `docker compose up --build`, THE SYSTEM SHALL start Postgres 16 with a named volume and a web container built from source that seeds and serves on port 8000.
- THE SYSTEM SHALL start the web container only after the database health check (`pg_isready`) passes.

## UI/UX overhaul

Added 2026-10-25 for the Full-track UI/UX overhaul (branch `sdlc/full-ui-overhaul`). These requirements come from the user's intake answers, recorded in `docs/decisions.md` (Intake rows). The user named three problems: "Looks generic/dated", "Too dense" and "Bad on mobile". The user did not pick "Hard to use", so the flows stay as they are (R-078). Tokens per second was dropped ("Drop it") and is out of scope.

Terms used below:

- *Phone width*: a 390 × 844 CSS px viewport. *Desktop width*: a 1280 × 800 CSS px viewport. These are the only two widths the designer mocks up and QA checks (Gate 1, Q-5).
- *Reference query*: `/results?vram_gb=24&context_tokens=4096`.
- *Result headline*: the statement of how many models fit or are tight (R-014), together with the hardware and context it applies to (R-022).
- *Rewordable text* (Gate 1, Q-2): headings, help text, notes and raw labels such as the breakdown's `weights_gb`. The designer may reword these and merge facts that appear twice. The approved new wording is the copy table in `docs/design/README.md` (section "Copy table (Q-2): what changed"), approved at Gate 2 (R-078).
- *Protected text* (Gate 1, Q-2): error messages, the verdict labels `Fits`, `Tight` and `Won't fit`, and every figure. These stay word for word and value for value.
- The numeric targets in this section (pixel offsets, tap size, text size, contrast) were accepted at Gate 1 (Q-4).

Requirements R-001 to R-066 stay in force unless a requirement in this section changes them. The overhaul changes how screens look and how they are laid out, not what the app computes, accepts or links to. The exceptions are the wording rules above, the two copy fixes in R-069, the `Cache-Control` header in R-080 and the history, reload and old-link fix in R-081.

### R-067 Screens and states in scope

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25 and Gate 2 approval (`docs/decisions.md`); `docs/design/README.md` ("Screens and states"); `docs/design/final/`, `docs/design/mockups/final-*.html`; `templates/base.html`, `index.html`, `results_summary.html`, `results_list.html`, `results_why.html`, `model_detail.html`, `how_it_works.html`

Asked: "UI/UX overhaul". Screens: "Whole app, incl. mobile" (offered as "every page, with the 390 px phone layout treated as a first-class design") and "How it works" (offered as "the /how-it-works explainer page"). Gate 2: "Approved the design: all 24 screens and states in `docs/design/final/` and `docs/design/mockups/final-*.html`, the copy table in `docs/design/README.md`, and ADRs 0001-0008. These are QA's reference (rule 10)."

- THE SYSTEM SHALL render these screens in the visual direction chosen at Gate 2, "D layout + E glass" (R-068):
  - the header and footer on every page;
  - `/` and `/results`: the input form, filters, error and GPU-search messages, summary, Top Picks, active filter chips, result count, model cards with quantisation switcher and fit gauge, empty state and unseeded state;
  - the "How this is estimated" breakdown loaded from `/results/why/<variant_id>`;
  - `/models/<model_id>`;
  - `/how-it-works`.
- WHEN QA opens any state below at phone width (390 px) and at desktop width (1280 px), in the light and the dark theme, THE SYSTEM SHALL match the approved mockup for that state (constitution rule 10; Gate 1, Q-5 and Q-11; Gate 2; R-068). The states are:
  1. `/` with no input, showing the assumed 8 GB note (R-005);
  2. a GPU selected;
  3. manual VRAM entered;
  4. ambiguous GPU search feedback (R-003, search text `RTX`);
  5. GPU not found, with a GB hint (`RTX 3060 12GB`);
  6. inline input errors (R-008, `vram_gb=abc`);
  7. active filter chips (R-031);
  8. Top Picks shown, and Top Picks hidden (R-024);
  9. a card whose open variant is `Tight`, with a struck-through `Won't fit` quantisation and a `Gated` tag;
  10. the breakdown before it loads, while it loads, and once it has loaded;
  11. the empty state, with `Try Q4` and `Lower context` both as links and both as inert text (R-025);
  12. the unseeded catalog (R-026);
  13. model detail for a gated model and for an ungated one, with and without sources, and with an inline VRAM error (R-033 to R-036);
  14. `/how-it-works`.
- Approved reference (Gate 2, 2026-10-25): for each name `<name>` below, the live mockup `docs/design/mockups/final-<name>.html` (add `?theme=dark` for the dark theme) and its four renders `docs/design/final/<name>-<390 or 1280>-<light or dark>.png`, 96 renders in all. The query each was built from is in `docs/design/README.md` ("Screens and states"). These 24 are the complete list QA checks; no other mockup is approved.

  | # | Approved mockup `<name>` | State | R-067 state |
  | --- | --- | --- | --- |
  | 1 | `home-default` | Home, no input | 1 |
  | 2 | `results-gpu` | A GPU selected | 2 |
  | 3 | `results-manual-vram` | Manual VRAM: the reference query | 3, 8 |
  | 4 | `results-gpu-ambiguous` | Ambiguous GPU search | 4 |
  | 5 | `results-gpu-notfound` | GPU not found, with a GB hint | 5 |
  | 6 | `results-errors` | Inline input errors | 6 |
  | 7 | `results-chips` | Active filter chips | 7 |
  | 8 | `results-filters-open` | Filters open | 7 |
  | 9 | `results-picks-hidden` | Top Picks hidden | 8 |
  | 10 | `results-tight-gated` | A Tight card, struck-through quantisations, Gated | 9 |
  | 11 | `gpu-suggest` | GPU suggestions (the styled list of R-071) | (R-071) |
  | 12 | `why-closed` | Breakdown before it loads | 10 |
  | 13 | `why-loading` | Breakdown loading | 10 |
  | 14 | `why-loaded` | Breakdown loaded | 10 |
  | 15 | `empty-links` | Empty state, both next steps are links | 11 |
  | 16 | `empty-inert` | Empty state, both next steps are inert | 11 |
  | 17 | `empty-mixed` | Empty state, one link and one inert | 11 |
  | 18 | `unseeded` | Unseeded catalog | 12 |
  | 19 | `detail-gated` | Gated model with sources | 13 |
  | 20 | `detail-ungated` | Ungated model with sources | 13 |
  | 21 | `detail-no-sources` | Model without sources | 13 |
  | 22 | `detail-error` | Model detail with an inline VRAM error | 13 |
  | 23 | `how-it-works` | How it works | 14 |
  | 24 | `focus-states` | Focus ring on every control | (R-077) |

- WHEN QA opens the state of any row above at phone width and at desktop width, in the light and the dark theme, THE SYSTEM SHALL match that row's approved mockup and renders.
- `docs/design/mockups/final-spec.html` is the designer's spec sheet, a document and not a screen; it is not one of the 24.

### R-068 One visual identity, applied everywhere

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answers Q-7 and Q-11, Gate 2 round 1 and round 1 redo (`docs/decisions.md`); `docs/design/` (direction boards D, E and F; round 2 mockups in `docs/design/final/`); `styles.css` (`.state` on cards, `.badge` on model detail; `.tag` vs `.badge-gated`), `model_detail.html` (unstyled `Recompute` button), `base.html` (text wordmark only, no favicon)

Asked: "Looks generic/dated" (offered as "no visual identity; it should look like a product someone designed"). Visual direction: "Designer proposes 2-3". Gate 1: "each direction includes a logo mark and favicon"; "Gate 2 runs in two rounds: 2-3 direction boards (home/results at 390 and 1280, both themes), then every screen and state for the chosen direction." Gate 2 round 1 redo: "Direction: "D layout + E glass": D Bento's tile grid built from E Glass's frosted panes and type (Outfit). Every screen and state is mocked up in this hybrid for Gate 2."

- Process at Gate 2 (not system behaviour), in two rounds (Q-11):
  - Round 1: the designer presents 2 or 3 distinct visual directions. Each is a board showing `/` and `/results` at phone width and at desktop width, in the light and the dark theme, and each includes a logo mark and a favicon (Q-7). The user picks one direction.
  - Round 2: every state in R-067 is mocked up in the chosen direction, at both widths and in both themes, and the user approves them.
- Chosen direction (Gate 2 round 1 redo, 2026-10-25): **"D layout + E glass"**, that is D Bento's tile grid built from E Glass's frosted panes and E Glass's type, Outfit. The user rejected directions A, B and C, and then picked this hybrid of D and E from directions D, E and F.
- Approved reference: the user approved the round 2 mockups at Gate 2 (2026-10-25). The 24 approved mockups listed in R-067 (`docs/design/final/` plus `docs/design/mockups/final-*.html`) are what every "approved mockup" in this section refers to.
- THE SYSTEM SHALL render every screen in R-067 in the direction "D layout + E glass", as the approved mockups in `docs/design/final/` show it, and SHALL NOT leave any screen in its pre-overhaul styling.
- WHEN QA compares the same kind of element on two screens, THE SYSTEM SHALL render both with equal computed font family, font size, font weight, text colour, background, border and corner radius. This covers:
  - primary buttons (`Check compatibility`, `Recompute`);
  - secondary buttons (`Clear`, `Apply filters`);
  - text, number and select fields;
  - chips;
  - disclosure controls;
  - links in body text;
  - classification labels.
- WHEN a variant is `Fits`, `Tight` or `Won't fit`, THE SYSTEM SHALL give its classification label the same visual treatment on a results card and on the model detail page. Today the two pages style it differently.
- WHEN a model is gated, THE SYSTEM SHALL give its gated marker the same visual treatment on a results card and on the model detail page. Today the two pages style it differently.
- THE SYSTEM SHALL NOT show any button, text field, number field, select, checkbox or radio on a screen in R-067 in the browser's default appearance. Each SHALL match the approved mockup.
- THE SYSTEM SHALL show the logo mark of the approved "D layout + E glass" mockups in the header of every page, as the approved mockup places it (Q-7).
- THE SYSTEM SHALL include in the `<head>` of every full HTML page a `<link rel="icon">` whose `href` is a file under `/static/` that returns HTTP 200, and that file SHALL be the favicon of the approved "D layout + E glass" mockups (Q-7). Serving it from `/static/` keeps the route list of R-042 unchanged.

### R-069 The answer leads the results

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answers Q-1, Q-2, Q-3, Q-4 and Q-10, Gate 2 approval (`docs/decisions.md`); `docs/design/README.md` (R-069 measurements, "Where a requirement pushed back", "Decisions for you" item 3, copy table); ADR-0006; `index.html` (filters `<details open>` between the form and the results), `results_summary.html`, `results_list.html` (count sentence after Top Picks and filter chips; `There are <n> variants in Won't fit` line), `main.py` (`_top_picks` descriptions); PR #19 (Top Picks copy deferred)

Asked: "Too dense" (offered as "too much information at once; the answer gets lost"). Gate 1: "the designer shows Top Picks placement options at Gate 2"; "filters start collapsed on a phone with an active-filter count"; "fix only the two copy quirks (empty state "variants" vs models, Top Picks copy)". Gate 2: "Filters start closed at both widths (not opened on desktop by script). The 800 px desktop target (R-069) applies to the normal results view; states that show a message (ambiguous GPU, input errors) may end around 840-880 px." The Gate 2 dispatch set the ceiling for those states at 900 px, added the GPU-not-found state to them, and kept 1688 px at phone width for every state.

- THE SYSTEM SHALL place the result headline first in the results area, before Top Picks, active filter chips and the cards. Today the hardware and the counts sit at the top, but the "N models fit or are tight" sentence comes after Top Picks and the filter chips.
- Process at Gate 2 (not system behaviour): the designer shows options for where Top Picks sit relative to the cards, and the user picks one (Q-1).
- THE SYSTEM SHALL place Top Picks where the mockup approved at Gate 2 places them (Q-1).
- In the criteria below, a position "ends within N CSS px" when the bottom edge of the element's box is at most N CSS px from the top of the page, measured on first load with the filters closed.
- WHEN the reference query loads at desktop width, THE SYSTEM SHALL end the result headline, and the first card's model name, verdict and margin, within the first 800 CSS px of the page, so they are visible without scrolling (Q-4).
- WHEN any of these message states loads at desktop width, THE SYSTEM SHALL end the result headline, and the first card's model name, verdict and margin, within the first 900 CSS px of the page (Gate 2):
  - ambiguous GPU search, `/results?gpu_search=RTX`;
  - GPU not found, `/results?gpu_search=RTX%203060%2012GB` (search text `RTX 3060 12GB`);
  - input errors, `/results?vram_gb=abc`.
- WHEN any state of `/` or `/results` that shows at least one model card loads at phone width, including the reference query and the three message states above, THE SYSTEM SHALL end the result headline, and the first card's model name, verdict and margin, within the first 1688 CSS px of the page, which is two phone screens (Q-4; Gate 2).
- WHEN `/` or `/results` loads at phone width or at desktop width, with or without JavaScript, THE SYSTEM SHALL show the filters closed, behind one disclosure control (Q-3; Gate 2).
- THE SYSTEM SHALL NOT open the filters by script on page load at any width (Gate 2).
- THE SYSTEM SHALL label that control `Filters (N)`, where N is the number of active filter chips (R-031): for example `Filters (2)` for `/results?vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4`.
- WHEN no filter is active, THE SYSTEM SHALL label that control `Filters (0)`.
- WHEN the user activates that control, THE SYSTEM SHALL show every filter control of R-027 to R-030, with or without JavaScript.
- THE SYSTEM SHALL render each fit pane's verdict and margin at a font size no smaller than any other text in that pane.
- THE SYSTEM SHALL NOT add any figure or text to a model card beyond what R-015 to R-021 specify.
- THE SYSTEM SHALL keep the summary's hardware label, context, default 8 GB note, `Fits`, `Tight` and `Won't fit` counts and GPU feedback (R-005, R-022, R-023), the active filter chips (R-031) and Top Picks (R-024), with the values, links and selection rules those requirements state. Where facts appear twice today (for example the `Fits`/`Tight` counts and the "N models fit or are tight" sentence), the designer may merge them (Q-2), but each value SHALL still appear at least once.
- WHEN the empty state shows and the `Won't fit` count is above 0, THE SYSTEM SHALL describe that count as models, not variants, because it counts models (R-023); the sentence SHALL NOT contain the word `variants`. Today it reads `There are <n> variants in Won't fit, hidden by default.` (Q-10 (i)). The approved wording (copy table) is `There are <n> models in Won't fit, hidden by default.`, and `There is 1 model in Won't fit, hidden by default.` when the count is 1.
- THE SYSTEM SHALL give each Top Pick a one-line description that says what the pick offers the user, not how the ranking works (Q-10 (ii)). No description SHALL contain `ranked` or `ranking`, and none SHALL equal today's `Highest-ranked variant that remains in the Fits band.`, `Next highest-ranked practical option by current ranking order.` or `Lowest-ranked viable option, usually the lightest footprint.` The approved wording (copy table) is `The strongest model that still leaves room to spare.` for `Best quality that fits`, `Nearly as capable, and comfortable to run.` for `Best balance`, and `The lightest model that still runs on your GPU.` for `Best small`. The selection rules of R-024 stay.

### R-070 Phone layout on every page

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answers Q-4 and Q-5 (`docs/decisions.md`); `styles.css` (breakpoints at 980, 780 and 720 px; `min-width: 44px` on quantisation buttons); `data/seed_models.json`, `data/seed_gpus.json` (longest names)

Asked: "Bad on mobile" (offered as "the layout breaks or is cramped on a phone"). Also "Whole app, incl. mobile" (offered as "the 390 px phone layout treated as a first-class design").

- WHEN any screen in R-067 is shown at phone width, THE SYSTEM SHALL NOT scroll horizontally: `document.documentElement.scrollWidth` SHALL equal 390. This SHALL hold:
  - in both themes;
  - with the breakdown open;
  - for cards showing the longest catalog names, `Mistral NeMo 12B Instruct (2407)` and `DeepSeek-Coder-V2-Lite-Instruct`;
  - with the longest GPU label, `NVIDIA GeForce RTX 4090 Laptop GPU (16.0 GB)`, in the GPU field and in the summary.
- THE SYSTEM SHALL NOT clip, overlap, or cut off with an ellipsis any text at phone width. Model names, GPU labels, source URLs and formulas SHALL wrap inside their container.
- THE SYSTEM SHALL give each of these a tap target of at least 44 × 44 CSS px at phone width, with no two targets overlapping (Q-4):
  - buttons, chips and quantisation buttons;
  - context presets;
  - checkboxes and radios, together with their labels;
  - header links and the theme toggle;
  - disclosure controls.

  Links inside running text are exempt.
- THE SYSTEM SHALL render every text field, number field and select at a font size of at least 16 CSS px at phone width, so focusing one does not zoom the page in iOS Safari.
- THE SYSTEM SHALL render no visible text smaller than 12 CSS px at any width (Q-4).
- WHEN a user on a phone focuses `Available VRAM (GB)`, `Min params (B)`, `Max params (B)` or `Manual VRAM (GB)`, THE SYSTEM SHALL bring up a numeric keyboard.
- WHILE at phone width, THE SYSTEM SHALL keep `Estimator`, `How it works` and the theme toggle reachable from the header with at most one tap before each.

### R-071 Input form on `/` and `/results`

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; `index.html` (form errors and `gpu_search_feedback` rendered in the form), `results_summary.html` (the same messages rendered again above the summary), `index.html` `hx-target="#results-shell"` (the form is not re-rendered on in-place refresh); Gate 1 answers Q-2, Q-4 and Q-8, Gate 2 approval (`docs/decisions.md`); ADR-0005 (`docs/adr/0005-gpu-suggestions-progressive-combobox.md`); approved mockup `gpu-suggest` (`docs/design/mockups/final-gpu-suggest.html`)

Asked: "Too dense"; "Bad on mobile"; "Looks generic/dated". Gate 1: "restyle the controls only, but the native GPU suggestion list may be replaced if it works badly on phones". Gate 2: "Build the styled GPU suggestion list (the `gpu-suggest` mockup): a small vanilla JS combobox over the native list at every width (ADR-0005), keeping R-001's behaviour and the no-JavaScript path."

- WHEN `/` loads at phone width or at desktop width, THE SYSTEM SHALL show `Search GPU`, `Available VRAM (GB)`, the three context presets and `Check compatibility` without scrolling (Q-4).
- THE SYSTEM SHALL keep both input paths, GPU search (R-001, R-003) and manual VRAM (R-002), the example chips (R-007) and the context presets (R-006), with the behaviour those requirements state.
- THE SYSTEM SHALL show the input error messages of R-008 word for word (protected text, Q-2). The form's headings and help text may change only to the wording of the copy table (R-078).
- Styled GPU suggestion list (Gate 2: chosen, not conditional). Every other control is restyled only (Q-8).
- WHILE JavaScript is enabled, at phone width and at desktop width, THE SYSTEM SHALL show the `Search GPU` suggestions in a styled list under the field, as the approved mockup `gpu-suggest` shows, in place of the browser's native suggestion list.
- WHILE the styled list is in use, THE SYSTEM SHALL NOT show the browser's native suggestion list for `Search GPU` at the same time.
- WHEN `Search GPU` is empty and the styled list opens, THE SYSTEM SHALL list every catalog GPU, in the order and with the labels R-001 states (VRAM ascending, then name; `<vendor> <name> (<vram to 1 decimal> GB)`).
- WHEN the user types into `Search GPU`, THE SYSTEM SHALL list only the GPUs whose label contains the typed text, compared case-insensitively with whitespace collapsed, in the order R-001 states. For example, typing `4060` lists `NVIDIA GeForce RTX 4060 (8.0 GB)`, `NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)`, `NVIDIA GeForce RTX 4060 Ti (8GB) (8.0 GB)` and `NVIDIA GeForce RTX 4060 Ti (16GB) (16.0 GB)`, in that order, with catalog `2026-10-23`.
- WHEN the user picks a suggestion by tap or click, THE SYSTEM SHALL put that suggestion's exact label in `Search GPU`, set that GPU as the selected GPU and empty `Available VRAM (GB)`, as R-001 states.
- WHEN the user moves through the suggestions with the Up and Down arrow keys and presses Enter, THE SYSTEM SHALL put the highlighted suggestion's exact label in `Search GPU`, set that GPU as the selected GPU and empty `Available VRAM (GB)`, as R-001 states.
- WHEN the user presses Escape while the styled list is open, THE SYSTEM SHALL close the list and keep the text in `Search GPU` unchanged.
- THE SYSTEM SHALL keep R-001's other behaviour with the styled list in use: a selected GPU is cleared when the text stops matching a suggestion exactly, and `Clear` empties the field, clears the selected GPU and moves focus to the field.
- THE SYSTEM SHALL give each suggestion in the styled list a tap target of at least 44 CSS px tall at phone width (R-070).
- THE SYSTEM SHALL expose the styled list to assistive technology with `role="combobox"` and `aria-expanded` on `Search GPU`, `role="listbox"` on the list and `role="option"` on each suggestion, as the approved mockup `gpu-suggest` marks it up (R-077).
- THE SYSTEM SHALL build the styled list in the site's own JavaScript, adding no third-party library, no new request and no new route (ADR-0005; R-042).
- WHILE JavaScript is disabled, THE SYSTEM SHALL show `Search GPU` with the native suggestion list of R-001, accept free text, and resolve it on submit as R-003 states.
- WHEN a full page load carries input errors (R-008), THE SYSTEM SHALL show the error list once, in an element with `role="alert"`. Today `/results?vram_gb=abc` shows it twice, once in the form and once above the summary.
- WHEN a full page load carries GPU search feedback (R-003), THE SYSTEM SHALL show the message once, in an element with `role="status"`. Today it appears both under the field and above the summary.
- WHEN results refresh in place, THE SYSTEM SHALL show only the error and feedback messages that belong to the current query. For example, after loading `/results?vram_gb=abc`, typing `24` and pressing `Check compatibility`, no error remains visible.

### R-072 Model cards, quantisation switcher and fit gauge

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; `results_list.html`, `styles.css` (`.state::before` marker shapes, hatched reserve band, `[data-oversize]` strike-through); PR #19

Asked: "Too dense"; "Bad on mobile"; "Looks generic/dated".

- THE SYSTEM SHALL keep each card's content, default quantisation, switcher behaviour, gauge geometry, captions and ordering as R-015 to R-019 and R-021 state, including the gauge's `role="img"` label and its server-rendered widths.
- WHILE at phone width, THE SYSTEM SHALL show all four quantisation buttons of a card on a single row.
- THE SYSTEM SHALL tell `Fits`, `Tight` and `Won't fit` apart by their text label plus one cue that is not colour (for example a shape or pattern), in both themes.
- THE SYSTEM SHALL keep the gauge's reserve band visible and distinct from the filled bar in both themes.
- THE SYSTEM SHALL keep a `Won't fit` quantisation button distinguishable from the pressed and unpressed buttons without relying on colour.

### R-073 "How this is estimated" breakdown

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answer Q-2 (`docs/decisions.md`); `results_why.html` (raw labels `weights_gb`, `kv_cache_gb`, `runtime_overhead_gb`, `reserve_gb`, `required_vram_gb`), `results_list.html` (`why-panel`, placeholder `Open to load the breakdown.`)

Asked: "Whole app, incl. mobile"; "Too dense"; "Looks generic/dated". Gate 1: "raw labels (such as `weights_gb`) may be reworded".

- THE SYSTEM SHALL keep the breakdown collapsed until the user opens it, fetch it once, and show `<quant label> on <model name>` with the weights, KV cache, runtime overhead, reserve and required VRAM, each to 2 decimals, computed as R-020 states.
- THE SYSTEM SHALL label the five values with the wording approved in the Gate 2 mockups, which may replace the raw labels `weights_gb`, `kv_cache_gb`, `runtime_overhead_gb`, `reserve_gb` and `required_vram_gb` (Q-2). The five figures SHALL stay unchanged.
- THE SYSTEM SHALL show a required VRAM in the breakdown equal to the `GB needed` caption of the same pane.
- WHILE the breakdown is loading, THE SYSTEM SHALL show a loading indication that is visibly different from the loaded breakdown.
- WHEN the breakdown is open at phone width, THE SYSTEM SHALL show all five values within the card's width, without horizontal scrolling.

### R-074 Model detail page

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answers Q-4 and Q-8 (`docs/decisions.md`); `model_detail.html` (seven-column table in `.table-scroll`; plain `<button>` `Recompute`)

Asked: "Whole app, incl. mobile"; "Too dense"; "Bad on mobile".

- THE SYSTEM SHALL keep the model facts, KV cache estimate, sources, gated notice, recompute form and variant list with the content and behaviour of R-033 to R-036. The recompute form's GPU dropdown, its hardware precedence (R-035) and its fields are restyled only (Q-8).
- THE SYSTEM SHALL show for every variant its label and bucket, the recommended marker, bits, required VRAM, available VRAM, margin to 2 decimals, and its classification label (R-034, constitution rule 16).
- WHEN the page is shown at phone width, THE SYSTEM SHALL show each variant's label, required VRAM, available VRAM, margin and classification without any horizontal scrolling, whether of the page or of a region inside it.
- WHEN `/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096` loads at phone width, THE SYSTEM SHALL place the model name and the first variant's classification within the first 1688 CSS px of the page (Q-4).
- THE SYSTEM SHALL place each variant's provenance (notes and sources) after that variant's figures and classification.

### R-075 How it works page

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answer Q-6, Gate 2 round 1 and round 1 redo (Q-12) (`docs/decisions.md`); `how_it_works.html` (policy version typed into the heading), `main.py` (`ESTIMATION_POLICY_VERSION = "v0.1"`); `src/vramsherpa/estimation.py` (`estimate_breakdown`); `data/seed_models.json` and `data/seed_variants.json` (catalog `2026-10-23`: `model_llama_31_8b_instruct`, params_b 8.0, kv_gb_per_1k_ctx 0.48, no `discovered` flag; `variant_model_llama_31_8b_instruct_q4`, bits_effective 4.5); `tools/catalog_sources.yaml` (the model is curated there); constitution rule 16

Asked: "How it works" (offered as "the /how-it-works explainer page"). Also "Looks generic/dated" and "Too dense". Gate 1: "restyle it and add plain-language terms plus a worked example (8B at Q4, 2048 tokens = 6.72 GB). Formulas are unchanged, so `ESTIMATION_POLICY_VERSION` stays v0.1. Reading of rule 16: the three change together only when the estimate itself changes." Gate 2 round 1: "The /how-it-works worked example uses a real catalog 8B model, so the page agrees with that model's card (about 7.33 GB at Q4, 2048 tokens), replacing the 6.72 GB test inputs." Gate 2 round 1 redo (Q-12): "if Llama 3.1 8B Instruct leaves the catalog, /how-it-works keeps the worked example from fixed inputs, unnamed ("an 8B model at Q4")."

The example model is Llama 3.1 8B Instruct (`model_llama_31_8b_instruct`). It is curated in `tools/catalog_sources.yaml`, not auto-discovered, so the weekly catalog refresh (R-054, R-059) does not add or remove it.

- THE SYSTEM SHALL keep on `/how-it-works`:
  - the inputs;
  - the four formulas of R-009;
  - the reserve formula of R-010;
  - the three classification rules of R-011;
  - the limitations statement of R-037.

  The constants SHALL stay the same (1.10, 1.0, 0.05, 0.75 and 0.10), and so SHALL the meaning of each formula and rule.
- THE SYSTEM SHALL state on `/how-it-works` an estimation policy version equal to the footer's `Estimation policy` value, which is `v0.1` today.
- THE SYSTEM SHALL NOT change `src/vramsherpa/estimation.py` or `ESTIMATION_POLICY_VERSION` as part of this overhaul. `ESTIMATION_POLICY_VERSION` stays `v0.1`. At Gate 1 the user read constitution rule 16 as: `estimation.py`, `/how-it-works` and the policy version change together only when the estimate itself changes, so restyling and explaining the page does not bump the version (Q-6).
- THE SYSTEM SHALL explain on `/how-it-works`, in plain language, each term the formulas of R-009 and R-010 use: `params_b`, `bits_effective`, `kv_gb_per_1k_ctx`, `context_tokens`, `weights_gb`, `kv_cache_gb`, `runtime_overhead_gb`, `required_vram_gb`, `available_vram_gb` and `reserve_gb` (Q-6). The explanations are the wording approved in the Gate 2 mockups.
- THE SYSTEM SHALL show on `/how-it-works` a worked example that names the catalog model `Llama 3.1 8B Instruct` and uses its Q4 variant with a 2048-token context. It SHALL state the inputs params_b = 8, bits_effective = 4.5, kv_gb_per_1k_ctx = 0.48 and context_tokens = 2048, and these steps, each figure to 2 decimals as `estimation.py` computes it (Q-6; Gate 2 round 1):
  - weights 4.95 GB (8 × 4.5 ÷ 8 × 1.10 = 4.95);
  - KV cache 0.98 GB (0.48 × 2048 ÷ 1000 = 0.98304);
  - runtime overhead 1.40 GB (1.0 + 0.05 × 8 = 1.40);
  - required VRAM 7.33 GB (4.95 + 0.98304 + 1.40 = 7.33304).
- WHEN QA opens `/results?vram_gb=24&context_tokens=2048&family=llama` and shows the Q4 variant on the `Llama 3.1 8B Instruct` card, THE SYSTEM SHALL show a `GB needed` caption equal to the worked example's required VRAM (`7.33 GB needed` with catalog `2026-10-23`).
- WHEN QA opens that card's "How this is estimated" breakdown for Q4 at 2048 tokens, THE SYSTEM SHALL show weights, KV cache, runtime overhead and required VRAM equal to the worked example's four steps (4.95, 0.98, 1.40 and 7.33 with catalog `2026-10-23`).
- WHEN QA opens `/models/model_llama_31_8b_instruct?vram_gb=24&context_tokens=2048`, THE SYSTEM SHALL show for the Q4 variant a required VRAM equal to the worked example's required VRAM.
- WHEN the seeded catalog gives that model or its Q4 variant a params_b, bits_effective or kv_gb_per_1k_ctx other than the values above, THE SYSTEM SHALL show the catalog's values in the worked example, with the figures `estimation.py` computes from them, so the example still equals the card.
- IF the catalog has no model `model_llama_31_8b_instruct`, or that model has no Q4 variant (including when the catalog is unseeded, R-026), THEN THE SYSTEM SHALL answer `/how-it-works` with HTTP 200, show every other section of the page, and show the worked example with the inputs params_b = 8, bits_effective = 4.5, kv_gb_per_1k_ctx = 0.48 and context_tokens = 2048 and the four steps above (4.95, 0.98, 1.40 and 7.33 GB). The example SHALL be labelled as an 8B model at Q4 (the user's words: "an 8B model at Q4") and SHALL NOT contain the text `Llama 3.1 8B Instruct` or any other model name (Q-12, settled at Gate 2 round 1 redo).
- WHEN the page is shown at phone width, THE SYSTEM SHALL show every formula, classification rule and worked-example step in full, wrapping where needed, without horizontal scrolling.
- THE SYSTEM SHALL give the page one `h1` and one `h2` per section; the sections SHALL include inputs, formula, reserve buffer, classification, worked example and limitations.

### R-076 Header and footer

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; Gate 1 answer Q-7 (`docs/decisions.md`); `base.html`

Asked: "Whole app, incl. mobile"; "Looks generic/dated". Gate 1: "each direction includes a logo mark and favicon".

- THE SYSTEM SHALL keep the header's `VRAM Sherpa` link and its `Estimator` and `How it works` links with `aria-current` (R-039), and the theme toggle with its labels, saved choice and pre-paint behaviour (R-040).
- THE SYSTEM SHALL show the logo mark (R-068) in the header, and following the logo mark or the `VRAM Sherpa` wordmark SHALL load `/`.
- THE SYSTEM SHALL keep the footer's `Catalog version`, `Estimation policy` and `Last updated` values on every page (R-038).
- WHEN a page is shown at phone width, THE SYSTEM SHALL show all three footer values in full, with none overlapping another.
- THE SYSTEM SHALL render the header and footer identically on every screen in R-067.

### R-077 Accessibility through the redesign

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25; `styles.css` (`:focus-visible` rules, `prefers-reduced-motion` blocks); R-008, R-016, R-018, R-022, R-039, R-040; PR #19 (contrast measurements); Gate 1 answer Q-4 (`docs/decisions.md`)

Asked: "UI/UX overhaul".

- THE SYSTEM SHALL give text a contrast ratio of at least 4.5:1 against its background, in both themes. Large text (at least 24 px, or at least 18.66 px bold) SHALL have at least 3:1.
- THE SYSTEM SHALL give these a contrast ratio of at least 3:1 against adjacent colours, in both themes (Q-4; today's borders measure 2.2:1 per PR #19): field borders, button outlines, the gauge bar, the reserve band and the classification markers.
- WHEN an interactive element receives keyboard focus, THE SYSTEM SHALL show a visible focus indicator on it, in both themes.
- WHILE the visitor's system requests reduced motion, THE SYSTEM SHALL NOT animate transitions.
- THE SYSTEM SHALL keep every control operable by keyboard alone, in the order the controls appear on screen.
- THE SYSTEM SHALL keep the ARIA roles and states the existing requirements specify:
  - `aria-pressed` on quantisation buttons (R-016) and on the theme toggle (R-040);
  - `role="img"` and its label on the gauge (R-018);
  - `role="alert"` on input errors (R-008);
  - `role="status"` on GPU feedback (R-022);
  - `aria-current="page"` in the navigation (R-039).

### R-078 Behaviour the overhaul keeps

Status: planned · Track: full · UI: yes · Source: intake 2026-10-25 (the user did not pick "Hard to use"); Gate 1 answers Q-2 and Q-8, Gate 2 approval (`docs/decisions.md`); `docs/design/README.md` (section "Copy table (Q-2): what changed" and its list "Word for word (R-078)"); constitution rules 11 and 16; R-001 to R-066

Asked: "UI/UX overhaul". The user did not name the flow as a problem. Gate 1: "headings, help text, notes and raw labels (such as `weights_gb`) may be reworded and repeats merged. Error messages, verdict labels and figures stay. New wording is approved in the Gate 2 mockups."; "restyle the controls only, but the native GPU suggestion list may be replaced if it works badly on phones". Gate 2: "Approved the design: all 24 screens and states …, the copy table in `docs/design/README.md`, and ADRs 0001-0008. These are QA's reference (rule 10)."; "Build the styled GPU suggestion list (the `gpu-suggest` mockup) …".

*Copy table*: the table under "Copy table (Q-2): what changed" in `docs/design/README.md`, approved at Gate 2. Each row gives an Old string (what the app rendered before the overhaul), a New string and a Where column. It is the wording reference for this change.

- WHEN either query below loads after the overhaul, THE SYSTEM SHALL show the same models in the same order as `main` at f2ad2d8, with the same verdicts and the same required VRAM, available VRAM and margin figures:
  - the reference query;
  - `/results?vram_gb=12&context_tokens=8192&family=qwen&quant_bucket=Q4`.
- THE SYSTEM SHALL keep every route, query parameter and fragment of R-013, R-032 and R-042, so a URL that worked before the overhaul shows the same results after it. The one change is R-081: `/results/partials/content?<query>` opened as a normal page load redirects to `/results?<query>`, which shows the same results on the full page.
- THE SYSTEM SHALL keep the inputs, filters and live refresh (R-001 to R-008, R-027 to R-032), the three context presets (R-006) and the model detail form with its GPU dropdown and hardware precedence (R-035), with the same controls, fields and steps; these are restyled only (Q-8). The one replacement is the styled GPU suggestion list of R-071, built over the native list, which stays as the no-JavaScript path.
- THE SYSTEM SHALL show the required VRAM, the available VRAM and the margin with every verdict (constitution rule 16).
- THE SYSTEM SHALL keep protected text word for word: every error message R-001 to R-066 specify, the verdict labels `Fits`, `Tight` and `Won't fit`, and every figure (Q-2).
- WHERE a string appears in the copy table's Old column, THE SYSTEM SHALL show the row's New wording in the place its Where column names, or remove the string where the New column says `(removed …)` or `(merged …)`.
- WHERE the copy table's New column adds a string that had no Old wording (for example `Loading the breakdown`, `Kept free, not part of the total` and the visually hidden headings `Recompute for other hardware` and `Variants`), THE SYSTEM SHALL show that string word for word in the place its Where column names.
- THE SYSTEM SHALL keep word for word every string the app rendered before the overhaul that is not in the copy table's Old column, including every error message, GPU search message, verdict label, figure, field label, button label, link text and tag that R-001 to R-066 specify and that the list "Word for word (R-078)" in `docs/design/README.md` names.
- Repeated facts may be merged as the copy table and the approved mockups (R-067) merge them, so long as each value still appears at least once (Q-2).
- WHILE JavaScript is disabled, THE SYSTEM SHALL still submit the form and render results, cards and gauges, and show `Apply filters` (R-013, R-018, R-032).
- WHILE JavaScript is disabled, THE SYSTEM SHALL let the user open any section the redesign collapses.

### R-079 New styles reach returning visitors

Status: planned · Track: full · UI: no · Source: constitution rule 21; R-041; `main.py` (`ASSET_VERSION = "20261024a"`)

Asked: "UI/UX overhaul".

- WHEN the overhaul ships, THE SYSTEM SHALL request its CSS and JS with an asset version other than `20261024a`.

### R-080 HTML responses carry `Cache-Control`

Status: planned · Track: full · UI: no · Source: Gate 1 answer Q-9 (`docs/decisions.md`); constitution rule 9; `main.py` (no route or middleware sets `Cache-Control` today; see "Other docs against the code" below)

Asked: "the overhaul ships with a `Cache-Control` header on HTML responses (a new UI: no requirement in this change)."

Check URLs, each requested with `GET` against a running app (for example `http://localhost:8000`) and a `Host` the allowlist accepts (R-043):

1. `/`
2. `/results?vram_gb=24&context_tokens=4096`
3. `/results?vram_gb=24&context_tokens=4096` with the request header `HX-Request: true`
4. `/results/partials/content?vram_gb=24&context_tokens=4096` with the request header `HX-Request: true` (without it, R-081 answers with a redirect)
5. `/results/partials/summary?vram_gb=24&context_tokens=4096`
6. `/results/partials/list?vram_gb=24&context_tokens=4096`
7. `/results/why/variant_model_qwen25_7b_instruct_q4?vram_gb=24&context_tokens=4096`
8. `/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096`
9. `/how-it-works`

QA can read the headers with `curl -s -o /dev/null -D - '<base><url>'` (add `-H 'HX-Request: true'` for URLs 3 and 4). The routes accept `GET` only, so use `-D -` rather than `curl -I`, which sends `HEAD`.

- WHEN a client requests any check URL, THE SYSTEM SHALL answer HTTP 200 with a `Content-Type` that starts with `text/html` and exactly one `Cache-Control` header.
- WHEN a client requests any check URL, THE SYSTEM SHALL send a `Cache-Control` value that contains the directive `no-cache` or the directive `no-store`, so the browser fetches or revalidates the HTML on every load and a deploy shows on the next page load (constitution rule 9).
- WHEN a client requests any check URL, THE SYSTEM SHALL NOT send a `Cache-Control` value that contains `immutable`, or a `max-age` or `s-maxage` above 0.

Out of scope: `/static/…` files, whose freshness comes from the asset version (R-041, R-079, constitution rule 21), and `/healthz` and the 400 and 404 responses, which are JSON.

### R-081 Back, forward, reload and shared links show the full page

Status: planned · Track: full · UI: no (no new screen or state: every page this requirement shows is the results state of R-067) · Source: Gate 2 round 1 and round 1 redo, Q-13 (`docs/decisions.md`); ADR-0008 (`docs/adr/0008-htmx-history-restores-full-page.md`); `index.html` (`hx-get="/results/partials/content"` with `hx-push-url="true"`, so each in-place refresh pushes a fragment URL); `main.py` (`results_partial_content` always returns `results_content.html`; `results` returns it for any `HX-Request: true`); `base.html` (htmx 1.9.12, default history cache)

Asked: "Fix the htmx history-miss bug (a back navigation replaces the page with the bare results fragment) in this change." Gate 2 round 1 redo: "Q-13 resolved by ADR-0008: reload and shared links show the full page, so R-081 covers reload and share as well as Back/Forward."; "Old saved or shared links to `/results/partials/content?…`, opened as a normal page, redirect to `/results?…`."

Today, each in-place refresh puts `/results/partials/content?<query>` in the address bar and in the browser history (R-032). When htmx has no cached copy of a history entry, it fetches that URL and puts the response in place of the page. The response is the results fragment, so the header, the form and the footer disappear. Reloading the page or opening the copied address shows the same bare fragment, without the stylesheet.

Test setup, in a real browser with JavaScript enabled, at desktop width and again at phone width:

- *Steps A*: open `/results?vram_gb=24&context_tokens=4096`, select the `8k` preset (the results refresh in place), then select the `2k` preset.
- *Steps B*: open `/results?vram_gb=24&context_tokens=4096` and tick the `qwen` and `llama` family checkboxes (the results refresh in place after each).
- *Forcing a history miss*: in the browser console run `localStorage.removeItem('htmx-history-cache')`. The 12-refresh criterion forces misses without the console.
- *Full page*: exactly one header with the `VRAM Sherpa` link, `Estimator`, `How it works` and the theme toggle; exactly one input form with `Search GPU`, `Available VRAM (GB)`, the context presets and `Check compatibility`; the filters; the summary and the model cards (or the empty state); and exactly one footer with `Catalog version`, `Estimation policy` and `Last updated`; all styled by the site stylesheet.
- *Old-form address*: `/results/partials/content?<query>`, the address an in-place refresh put in the address bar before this change, as found in old bookmarks, shared links and browser history.

Back and forward:

- WHEN the user completes Steps A, forces a history miss and presses Back, THE SYSTEM SHALL show the full page, not the results fragment alone.
- WHEN the user presses Back after a history miss as above, THE SYSTEM SHALL show results and form state for that history entry: the `8k` preset selected and the summary's context stating `8k`, as `/results?vram_gb=24&context_tokens=8192` shows it when opened directly.
- WHEN the user then forces another history miss and presses Forward, THE SYSTEM SHALL show the full page with the `2k` preset selected and the summary's context stating `2k`.
- WHEN the full page is restored after a history miss, THE SYSTEM SHALL NOT show a second header, form or footer nested inside the page.
- WHEN the user changes a context preset on a page restored after a history miss, THE SYSTEM SHALL refresh the summary and results in place without a full page load, as R-032 states.
- WHEN the user presses Back or Forward without a history miss (the cache is left alone), THE SYSTEM SHALL show the full page for that history entry, as it does today.
- WHEN the user makes 12 in-place refreshes from `/results?vram_gb=24&context_tokens=4096` (alternating the `2k` and `8k` presets) and then presses Back 12 times, THE SYSTEM SHALL show the full page after every press.

Address bar, reload and shared links:

- WHEN results refresh in place, THE SYSTEM SHALL put in the address bar an address whose path is `/results` and whose query string is the query string of that refresh's request. After Steps A the address SHALL have the path `/results` and carry `vram_gb=24` and `context_tokens=2048`.
- WHEN the user completes Steps A and reloads the page, THE SYSTEM SHALL show the full page with the `2k` preset selected, the summary's context stating `2k`, and the same models in the same order with the same verdicts and figures as before the reload.
- WHEN the user completes Steps B and reloads the page, THE SYSTEM SHALL show the full page with `qwen` and `llama` ticked and the same models in the same order with the same verdicts and figures as before the reload.
- WHEN the user completes Steps B, copies the address from the address bar and opens it in a new private window (empty cache and storage), THE SYSTEM SHALL show the full page with `qwen` and `llama` ticked and the same models in the same order with the same verdicts and figures as the original tab.

Old-form addresses:

- WHEN a request to `/results/partials/content` does not carry `HX-Request: true` (a normal page load), THE SYSTEM SHALL answer with an HTTP redirect, status 302, 303 or 307, whose `Location` header is `/results?<query>`, where `<query>` is the request's query string unchanged, byte for byte, with repeated `family` and `quant_bucket` parameters kept in their order.
- WHEN such a request has no query string, THE SYSTEM SHALL send `Location: /results`.
- THE SYSTEM SHALL send that `Location` as a path with no scheme and no host, so behind the tunnel the browser stays on `https` (constitution rule 8, R-045).
- THE SYSTEM SHALL NOT answer that redirect with status 301 or 308. Browsers cache permanent redirects, and a cached one would also catch the in-place refreshes, which still request `/results/partials/content` with the same queries.
- WHEN a browser opens `/results/partials/content?vram_gb=24&context_tokens=4096&family=qwen&family=llama` as a normal page load, THE SYSTEM SHALL end at the address `/results?vram_gb=24&context_tokens=4096&family=qwen&family=llama` and show the full page with the same models in the same order, with the same verdicts and figures, as that `/results` address opened directly.
- WHEN the user has opened an old-form address as above and then selects the `8k` preset, THE SYSTEM SHALL refresh the summary and results in place without a full page load, as R-032 states.
- WHEN a request to `/results/partials/content`, `/results/partials/summary` or `/results/partials/list` carries `HX-Request: true`, THE SYSTEM SHALL answer it as it does today: HTTP 200 with the same fragment, and no redirect.
- THE SYSTEM SHALL NOT redirect requests to `/results/partials/summary` or `/results/partials/list`, with or without `HX-Request: true`.

## Notes

- Not recorded as requirements: branch `feat/tokens-per-second` (ee788b3) is unmerged work in progress. PR #19 lists three deliberately deferred items: Top Picks copy, raw Hugging Face names on discovered models, and tokens per second. At Gate 1 the Top Picks copy came into scope (R-069); the other two stay out (Q-10; tokens per second was dropped at intake).
- Quirks seen in the code, recorded here only so they are not lost: (a) when the hardware comes from an unmatched GPU search with a GB hint (R-003), the card's model link and breakdown pass `gpu_search`, which `/models/…` and `/results/why/…` ignore, so they compute for 8 GB instead of the hinted value. (b) The empty-state sentence says "variants" but the number counts models (R-023, R-025). (c) Hardware precedence is opposite on the two pages: `vram_gb` wins on `/results` (R-004) and `gpu_id` wins on model detail (R-035). (d) `results_table.html` is not rendered by any route. At Gate 1 only (b) came into scope (R-069, Q-10); (a), (c) and (d) stay out.
- Known issue, left out of this change by the user at Gate 2 round 1 redo (`docs/decisions.md`: "Not selected: re-wiring the theme toggle and GPU input after htmx restores a page from its cache. That stays a known issue, outside this change."): when Back or Forward restores a page from htmx's history cache (a cache hit, not the miss R-081 fixes), the theme toggle and the GPU input stop working on that page, because htmx swaps in the cached page without rerunning `app.js`'s setup. The quantisation switcher keeps working. ADR-0008 records the cause. No requirement covers it, and R-081's criteria do not test it.

## Contradictions resolved at Gate 0

`agent.md` (last changed 2026-02-22; archived at Gate 0 to `docs/archive/agent.md`, still gitignored) against the app. The user ruled the code right in every row (2026-10-25):

| What the file says | What the code does |
| --- | --- |
| `agent.md` Non-goals: "No external syncing (Hugging Face / Ollama registry) in MVP". | The weekly refresh queries the Hugging Face API to discover models, and optionally to enrich metadata (`tools/ingest_catalog.py` `_discover_models_from_hf`; `tools/catalog_sources.yaml` `discovery.enabled: true`). It runs at build time, not at request time. |
| `agent.md` Non-goals: "catalog is curated and small". | 95 models, 81 of them auto-discovered rather than curated, including models up to 1650B parameters (`data/seed_models.json`; `catalog_sources.yaml` `params_b_overrides`). |
| `agent.md` GET `/`: "GPU dropdown (from DB) OR manual VRAM input … Submit to results". | `/` is a free-text GPU search with suggestions, fuzzy matching and example chips, and it already renders full results for an assumed 8 GB (`index.html`; `main.py` `home`, `_resolve_available_vram`). |
| `agent.md` GET `/results`: "Show ranked list of model variants" with quant label, bits_effective, required, margin and a badge per variant. | Results show one card per model with a quantisation switcher. Models with no usable variant are hidden, cards do not show bits_effective, and margin appears as "X GB spare/over" (`results_list.html`; `main.py` `_group_results_by_model`; PR #19). |
| `agent.md` Sorting: "recommended first, then larger params_b, then higher bits_effective (within classification)". | Cards sort by the open variant's classification, then params_b descending, then name. bits_effective plays no part (`main.py` `_group_results_by_model`). The agent.md order now drives only Top Picks (`_build_results`). |
| `agent.md` GET `/healthz`: "Returns 200 and minimal JSON". | Returns 503 when the catalog is unseeded or the DB is down. The JSON carries counts, catalog_version and, on DB failure, the raw exception text (`main.py` `healthz`). |
| `agent.md` CI: "must run on pull_request and push to main". | `ci.yml` runs on `pull_request` and `workflow_call`. Pushes to main run it through `release.yml`. The effect is the same, but the trigger is not. |

Other docs against the code:

| What the file says | What the code does |
| --- | --- |
| `docs/RELEASE_AND_DEPLOY.md`: "`PUBLIC_HOSTNAME` must match the DNS name exactly. Because `APP_ENV=prod`, any other `Host` is rejected with 400"; the smoke test checks "`APP_ENV=prod` rejects an unknown `Host`". | The app never reads `PUBLIC_HOSTNAME`, and `APP_ENV` does not affect host checks. Only `ALLOWED_HOSTS` does (`src/vramsherpa/config.py`; `main.py` `host_guard`). With the documented `.env` (`ALLOWED_HOSTS=localhost,127.0.0.1,web`), requests for the public hostname would get HTTP 400. |
| `docs/RELEASE_AND_DEPLOY.md`: "A broken image therefore never gets tagged or released, so `latest` never moves to it." | "Build and push" pushes the `latest` tag before the smoke test runs (`.github/workflows/release.yml`). A failing smoke test blocks the git tag and the GitHub Release, but `:latest` has already moved, and watchtower can pull it. |
| `docs/RELEASE_AND_DEPLOY.md`: the smoke test checks that "the results page renders". | It requests `/?vram_gb=24&context_tokens=4096`. `/` ignores query parameters, so the test renders the home page at the 8 GB default and never calls `/results` (`release.yml`; `main.py` `home`). |
| `docs/constitution.md` rule 9: HTML responses send a `Cache-Control` header. | No route or middleware sets `Cache-Control` (no match in `src/`). Only the CSS and JS URLs carry a version query (R-041). R-080 plans the fix (Gate 1, Q-9). |
| `docs/constitution.md` rule 3: app ports bind to `127.0.0.1`. | Production complies (`127.0.0.1:8000`). The local `docker-compose.yml` publishes `8000:8000` on all interfaces. |

## Resolved at Gate 1

The user answered Q-1 to Q-11 at Gate 1 on 2026-10-25. The answers are recorded in `docs/decisions.md` (rows dated 2026-10-25, Gate 1) and folded into the requirements above. No open questions remain.

| # | Topic | Answer, in short | Folded into |
| --- | --- | --- | --- |
| Q-1 | Top Picks placement | The designer shows placement options at Gate 2. | R-069 |
| Q-2 | Wording and repeats | Headings, help text, notes and raw labels may be reworded and repeats merged; error messages, verdict labels and figures stay; wording approved in the Gate 2 mockups. | Terms, R-069, R-071, R-073, R-075, R-078 |
| Q-3 | Filters on a phone | Collapsed on first load, with an active-filter count. | R-069 |
| Q-4 | Numeric targets | Accepted as proposed: 1688 px phone and 800 px desktop to the first card, 44 px taps, 12 px minimum text, 3:1 non-text contrast. | R-069, R-070, R-071, R-074, R-077 |
| Q-5 | Widths | 390 px and 1280 px only. | Terms, R-067, R-070 |
| Q-6 | /how-it-works and rule 16 | Restyle, add plain-language terms and a worked example (6.72 GB, replaced at Gate 2 round 1 by the catalog model's 7.33 GB); formulas unchanged, policy stays v0.1; rule 16 applies only when the estimate changes. | R-075 |
| Q-7 | Logo and favicon | Each direction includes a logo mark and a favicon. | R-068, R-076 |
| Q-8 | Controls | Restyle only; the native GPU suggestion list may be replaced if it works badly on phones. | R-071, R-074, R-078 |
| Q-9 | Constitution rule 9 | Add a `UI: no` requirement for `Cache-Control` on HTML to this change. | R-080 |
| Q-10 | Known quirks | Only the two copy fixes: empty state "variants" vs models, and Top Picks copy. | R-069, Notes |
| Q-11 | Gate 2 shape | Two rounds: 2-3 direction boards (home and results at 390 and 1280, both themes), then every screen and state in the chosen direction. | R-067, R-068 |

## Open questions

None open. Q-12 and Q-13 were raised while applying the Gate 2 round 1 decisions and closed by the user at Gate 2 round 1 redo (2026-10-25); the answers are in `docs/decisions.md` (rows dated 2026-10-25, Gate "2 (round 1 redo)"). Neither conflicts with `docs/constitution.md`. Applying the Gate 2 approval (R-067, R-068, R-069, R-071, R-078) raised no new question, and none of those changes conflicts with the constitution; the styled GPU suggestion list adds JavaScript under `static/`, so rule 21 applies through R-079.

| # | Requirement | Question | Status and answer |
| --- | --- | --- | --- |
| Q-12 | R-075 | If `Llama 3.1 8B Instruct` or its Q4 variant ever leaves the catalog, should `/how-it-works` keep the worked example with the last known inputs (unnamed), or drop the worked example section? | Closed 2026-10-25. Keep it, unnamed ("an 8B model at Q4"), from the fixed inputs params_b 8, bits_effective 4.5, kv_gb_per_1k_ctx 0.48 and 2048 tokens (7.33 GB). Folded into R-075. See `docs/decisions.md`. |
| Q-13 | R-081, R-032 | Each in-place refresh puts `/results/partials/content?…` in the address bar, so reloading the page or sharing that address also shows the bare results fragment. The decision names only back navigation. Should R-081 also cover reload and opening the pushed address directly (which R-032's "reproduces the same results when opened directly" would suggest)? | Closed 2026-10-25, resolved by ADR-0008. Yes: reload and shared links show the full page, and old `/results/partials/content?…` links opened as a normal page redirect to `/results?…`. Folded into R-081; R-032's as-built note corrected. See `docs/decisions.md`. |
