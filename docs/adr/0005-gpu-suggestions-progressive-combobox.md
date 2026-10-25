# ADR-0005: Keep the native GPU datalist; if it fails on phones, add a combobox on top of it

Date: 2026-10-25 · Status: proposed (conditional on the Gate 2 finding) · Requirements: R-001, R-003, R-007, R-070, R-077, R-078

## Context

The GPU field on `/` is `<input id="gpu_search" list="gpu-options">` with a server-rendered `<datalist>` of all 35 catalog GPUs. Each option has a `data-gpu-id`. `app.js` (`initGpuInputs`, `initExampleChips`) copies the id into the hidden `gpu_id` field when the text exactly equals an option, and clears it otherwise (R-001). Without JavaScript the form still submits `gpu_search`, and the server resolves it (R-003). How the native datalist looks and behaves on phones is up to the browser: iOS Safari shows suggestions as a strip above the keyboard, and the list cannot be styled to match a direction. At Gate 1 (Q-8) the user kept every control but allowed replacing the native suggestion list if it works badly on phones. R-078 requires the same controls, fields and steps, and a working form without JavaScript.

## Decision

The `<datalist>` stays in the markup as the single source of suggestions and as the no-JavaScript and failure fallback. If the Gate 2 review finds that it works badly at 390 px (checked on iOS Safari and Android Chrome, and recorded in `docs/decisions.md`), `app.js` gains a small vanilla combobox following the WAI-ARIA 1.2 combobox pattern, applied at every width so there is one behaviour to test. On load it reads the options from `#gpu-options`, removes the input's `list` attribute so the native list and the custom one never both appear, and renders a `role="listbox"` popup under the field. The input gets `role="combobox"`, `aria-expanded`, `aria-controls` and `aria-activedescendant`. Filtering is a case-insensitive substring match with whitespace collapsed (as in R-003), in the catalog's order (VRAM ascending, then name, R-001). With an empty field the popup lists every GPU, with the same labels as the datalist (R-071). Options are at least 44 px tall (R-070), and the keys are Up, Down, Enter, Escape and Tab. Choosing an option writes its exact label into the field and dispatches `input`, so the existing exact-match logic sets `gpu_id` unchanged. `Clear`, the example chips and the manual VRAM field keep their current code paths. No library, request, route or template data is added. If the review finds the native list acceptable, this component is not built and the ADR is marked rejected.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Replace the datalist with a third-party autocomplete (Choices.js, Tom Select, Awesomplete) | Adds a dependency loaded from a CDN or vendored, plus its CSS to override with the direction's tokens, for 35 items that a short vanilla script handles. |
| A server-backed search (HTMX `hx-get` to a suggestion endpoint) | Needs a new route (R-042) and a request per keystroke, for data already in the page. |
| A `<select>` of GPUs on `/` | Changes the control and its steps (R-078, Q-8) and drops free-text search with VRAM hints (R-003). |
| Apply the combobox only below a phone breakpoint | Two behaviours to build, test and keep consistent (R-068) on the same field. |
| Keep the native datalist unconditionally | Acceptable if it works, which is why this ADR is conditional. Gate 1 allowed replacing it only on evidence that it works badly. |

## Consequences

- If the combobox is built, it is the largest JavaScript addition in the overhaul (about 120 to 150 lines in `app.js`), and it needs keyboard and screen-reader checks in QA (R-077) on top of the visual comparison (rule 10).
- With JavaScript off, or if the script fails, the native datalist and the server-side resolution behave exactly as today.
- The hidden `gpu_id` contract, element ids (`gpu_search`, `gpu_id`, `gpu-options`, `clear-gpu`) and the `.chip-example` data attributes are frozen for the overhaul, because both the old and the new script depend on them.
- The model detail page's GPU `<select>` is unchanged (Q-8 option (c) was not chosen).
