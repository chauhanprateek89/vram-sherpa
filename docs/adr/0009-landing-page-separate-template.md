# ADR-0009: `/` renders its own landing template and computes no results

Date: 2026-10-28 · Status: proposed · Requirements: R-083, R-001, R-002, R-005, R-006, R-007, R-013, R-026, R-039, R-042, R-067 (L1 to L4), R-071, R-076

## Context

`home` in `main.py` builds the default results today: `_resolve_available_vram` with no input (8 GB), `_build_results` at 2048 tokens, then `_results_context`, and renders `index.html`, the results page with the full form, filters, summary, Top Picks and cards. `results` renders the same template with the request's parameters.

R-083 turns `/` into a landing page: the header, one centred box (`Search GPU`), the three example chips, one submit control (`Check compatibility`) and the footer. It shows no summary, Top Picks, cards, filters, `Available VRAM (GB)`, context presets, default note, messages or `Clear`, and exactly one text field. It ignores query parameters and answers 200 (R-013, Q-19). With an unseeded catalog it shows the box and R-026's two sentences (Q-22). No route is added (R-042, rule 11).

## Decision

1. **A new template, `templates/landing.html`,** extends `base.html`, so the header (logo, `Estimator` with `aria-current`, `How it works`, theme toggle) and footer are the same markup as on every other page (R-076). It sets its own `main_class` (for example `landing`) so the stylesheet can centre the box without touching the bento layout. Its `content` block holds, in order: the `h1`, the help line, the landing form, the chips and, when the catalog is unseeded, the status block. It never includes `results_content.html`, the filters or `#results-shell`.
2. **The landing form** is `<form id="landing-form" method="get" action="/results">` with no `hx-*` attribute (ADR-0012). It holds:
   - `<label for="gpu_search">Search GPU</label>`, which may carry the existing `.sr-only` class (Q-20);
   - the `.combo` wrapper with `<input id="gpu_search" name="gpu_search" type="text" list="gpu-options" autocomplete="off" autofocus data-submit-on-choose>` and today's placeholder until Gate 2 rewords it;
   - the `<datalist id="gpu-options">`, from a new include shared with `index.html` (for example `templates/gpu_options.html`), so both pages emit the same 35 options, labels and `data-gpu-id` values that the combobox and R-001 rely on;
   - one `<button type="submit">` whose accessible name is `Check compatibility`, from visible text or from an `.sr-only` span beside an `aria-hidden` icon (Q-17).
   It has no `gpu_id`, `vram_gb`, `context_tokens` or `clear-gpu` element. The ids `gpu_search`, `gpu-options` and `.combo` are reused on purpose: `initGpuCombobox` finds them by those names and needs no change to start (ADR-0005).
3. **The chips** sit under the box in `<div class="chips" role="group" aria-label="Examples">` as plain links, `<a class="chip" href="…">`, without the `.chip-example` class or `data-*` attributes, so `initExampleChips` ignores them. Their `href` is built on the server from `_example_chips(gpus)` with `_results_url` (ADR-0012): `/results?gpu_id=<id>&context_tokens=2048` for the GPU chip, `/results?vram_gb=8&context_tokens=2048` and `/results?vram_gb=24&context_tokens=2048` for the others.
4. **`home(request, session)`** keeps its signature (tests call it directly). It builds `_base_context` plus `gpus = _load_gpus(session)` and the landing chips, and renders `landing.html`. It does not call `_resolve_available_vram`, `_build_results` or `_results_context`, so `/` runs no estimation. It declares no query parameters, so FastAPI ignores any it gets, including malformed ones such as `/?context_tokens=abc`, and the page is the same with or without them.
5. **Unseeded catalog.** `_base_context` already sets `catalog_seeded`. When it is false, `landing.html` shows under the box an element with `role="status"` holding `Catalog is not initialized yet.` and `Seed the catalog data, then refresh this page.`, answering 200 (R-026, Q-22). The `gpus` table is then empty, so the datalist has no options and the combobox returns early and leaves a plain text box. The two VRAM chips stay links. The GPU chip has no GPU to bind (R-007's fallback needs at least one GPU), so it renders as an inert `<span class="chip" aria-disabled="true">Example GPU</span>` rather than a link. Production always seeds before uvicorn starts (rule 12), so this state is local only.
6. **`index.html` stays the results page** under its current name. `results` and the partial routes keep rendering it and `results_content.html` as today.

## Alternatives considered

| Option | Why not |
| --- | --- |
| A landing mode inside `index.html` (`{% if landing %}` around the form and results) | One template with two unrelated layouts, every block guarded by a flag, and a missed guard leaks results markup, a second text field or `#results-shell` onto `/`. A separate template makes "no results on `/`" structural. |
| Rename `index.html` to `results.html` while adding `landing.html` | Clearer names, but it churns every test and ADR that names `index.html` for no behaviour change. Can be done later on its own. |
| Keep computing the 8 GB results in `home` and hide them with CSS | Wastes a full estimation on every landing view, and the hidden markup still counts for R-083's "exactly one text field" and for assistive technology. |
| Serve the landing page at a new path and redirect `/` to it | Adds a route (R-042, rule 11) and changes the address the user sees. |

## Consequences

- `/` costs two small queries (catalog version and the GPU list) instead of a full results build.
- `/?vram_gb=24&context_tokens=4096`, which the release smoke test requests, now renders the landing page. The smoke test adds a `/results` render check (ADR-0013).
- Tests that use `/` as a results or form page change to a `/results?…` query or to landing assertions (HLD, "Centred landing box", test strategy).
- The landing page carries the same `base.html` head, so the asset, favicon, font and `https` checks the smoke test runs on `/` keep covering every static URL.
- `landing.html` and the shared datalist include are new templates. Styling them changes `styles.css`, which bumps `ASSET_VERSION` (rule 21).
- `UI: yes`. The designer's L1 to L4 mockups decide the look; this ADR fixes the structure, ids and accessible names they must keep.
