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
- WHEN results refresh in place, THE SYSTEM SHALL update the browser address to a URL that reproduces the same results when opened directly.
- WHEN a request to `/results` carries `HX-Request: true`, THE SYSTEM SHALL return only the results fragment.
- THE SYSTEM SHALL serve fragments at `/results/partials/content`, `/results/partials/summary` and `/results/partials/list`, accepting the same parameters as `/results`.
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

## Notes

- Not recorded as requirements: branch `feat/tokens-per-second` (ee788b3) is unmerged work in progress. PR #19 lists three deliberately deferred items: Top Picks copy, raw Hugging Face names on discovered models, and tokens per second.
- Quirks seen in the code, recorded here only so they are not lost: (a) when the hardware comes from an unmatched GPU search with a GB hint (R-003), the card's model link and breakdown pass `gpu_search`, which `/models/…` and `/results/why/…` ignore, so they compute for 8 GB instead of the hinted value. (b) The empty-state sentence says "variants" but the number counts models (R-023, R-025). (c) Hardware precedence is opposite on the two pages: `vram_gb` wins on `/results` (R-004) and `gpu_id` wins on model detail (R-035). (d) `results_table.html` is not rendered by any route.

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
| `docs/constitution.md` rule 9: HTML responses send a `Cache-Control` header. | No route or middleware sets `Cache-Control` (no match in `src/`). Only the CSS and JS URLs carry a version query (R-041). |
| `docs/constitution.md` rule 3: app ports bind to `127.0.0.1`. | Production complies (`127.0.0.1:8000`). The local `docker-compose.yml` publishes `8000:8000` on all interfaces. |
