# ADR-0005: A styled GPU suggestion list, built as a combobox on top of the native datalist

Date: 2026-10-25 · Revised: 2026-10-25 (Gate 2) · Status: accepted · Requirements: R-001, R-003, R-007, R-068, R-070, R-071, R-077, R-078

> **Revision, 2026-10-25 (Gate 2).** The first version kept the native `<datalist>` and made the combobox conditional on evidence, found at Gate 2, that the native list works badly on phones. The designer could not test iOS Safari or Android Chrome, so no such evidence exists. At Gate 2 the user chose the styled list anyway ("Build the styled GPU suggestion list (the `gpu-suggest` mockup): a small vanilla JS combobox over the native list at every width (ADR-0005), keeping R-001's behaviour and the no-JavaScript path", `docs/decisions.md`). The combobox is now part of the build, not conditional. The mechanism is unchanged. The markup below now follows the approved mockup `docs/design/mockups/final-gpu-suggest.html`.

## Context

The GPU field on `/` is `<input id="gpu_search" list="gpu-options">` with a server-rendered `<datalist>` of all 35 catalog GPUs. Each option has a `data-gpu-id`. `app.js` (`initGpuInputs`, `initExampleChips`) copies the id into the hidden `gpu_id` field when the text exactly equals an option, and clears it otherwise (R-001). Without JavaScript the form still submits `gpu_search`, and the server resolves it (R-003). The browser decides how the native list looks and behaves: iOS Safari shows suggestions as a strip above the keyboard, desktop browsers draw their own grey popup, and none of it can be styled. R-068 asks that no control on a screen in R-067 show in the browser's default appearance, and the native popup is the one control the stylesheet cannot reach. At Gate 1 (Q-8) the user allowed replacing the native list. At Gate 2 the user approved the `gpu-suggest` mockup and chose to build it. R-078 still requires the same controls, fields and steps, and a working form without JavaScript.

## Decision

The `<datalist>` stays in the server-rendered markup, with `list="gpu-options"` on the input, as the single source of suggestions and as the no-JavaScript and failure fallback. `app.js` adds a small vanilla combobox that follows the WAI-ARIA 1.2 combobox pattern, at every width, so there is one behaviour to build and test.

1. **Start-up.** On `DOMContentLoaded` the script reads the options from `#gpu-options` (label from `value`, id from `data-gpu-id`). It removes the input's `list` attribute, so the native list and the custom one never both appear. It sets `role="combobox"`, `aria-autocomplete="list"`, `aria-expanded="false"` and `aria-controls="gpu-listbox"` on the input. It inserts `<ul id="gpu-listbox" class="listbox" role="listbox" aria-label="GPU suggestions" hidden>` inside the input's `.combo` wrapper (the template renders the wrapper; the script never moves the input). If any step fails, the script returns early and leaves the native list in place.
2. **Filtering and order.** A case-insensitive substring match with whitespace collapsed, as in R-003, over the catalog's order (VRAM ascending, then name, R-001). With an empty field the popup lists every GPU. Labels are exactly the datalist labels (R-071), for example `NVIDIA GeForce RTX 4060 (8.0 GB)`. Options are `<li id="gpu-opt-N" role="option">`, at least 44 px tall (R-070), and the active one carries `aria-selected="true"` and is named by the input's `aria-activedescendant`.
3. **Opening and closing.** The popup opens on input and on Down or Alt+Down. It closes on Escape (which keeps the typed text), on Tab (which moves focus on as normal), on blur, and when the text matches nothing. `aria-expanded` follows it.
4. **Keys.** Up and Down move the active option (no wrap past the ends). Enter with an active option chooses it and calls `preventDefault`, so it does not submit the form. Enter with the popup closed or no active option submits the form as today. Home and End stay with the text field.
5. **Choosing.** A tap, a click (handled on `pointerdown` with `preventDefault`, so the input keeps focus and blur does not close the popup first), or Enter writes the option's exact label into the field and dispatches `input` and `change`. The existing exact-match code then sets `gpu_id` and empties `Available VRAM (GB)` unchanged (R-001, R-071). `gpu_search` has no `js-auto-filter` class, so this does not trigger an in-place refresh. Choosing does not submit.
6. **What does not change.** `Clear`, the example chips (`.chip-example` with `data-gpu-id` and `data-vram-gb`) and the manual VRAM field keep their code paths. No library, request, route or template data is added. The model detail page's GPU `<select>` is unchanged.
7. **Style.** The `.listbox` rules come from `docs/design/build/components.css` (the "GPU suggestion popover" section). Below 900 px the popup uses the solid panel colour without blur (the glass budget in `docs/design/README.md`). The popup is positioned under the field inside `.combo` and is never wider than the field row, so it adds no horizontal scroll at 390 px (R-070). Long labels such as `NVIDIA GeForce RTX 4090 Laptop GPU (16.0 GB)` wrap inside the option.

Guards. pytest (rule 19) checks the server markup: the input keeps `list="gpu-options"` and the frozen ids, the datalist has all 35 GPUs in R-001's order with their labels and `data-gpu-id`, and the input sits inside a `.combo` wrapper. The existing `node --check` test covers the script's syntax. Keyboard, pointer and screen-reader behaviour at 390 and 1280 px is QA's check in a real browser (rule 10, R-077), against `final-gpu-suggest`.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Keep the native datalist (the first version of this ADR) | Not chosen at Gate 2. Its popup is browser chrome that cannot match the direction (R-068), and it looks different in every browser. |
| Replace the datalist with a third-party autocomplete (Choices.js, Tom Select, Awesomplete) | Adds a dependency loaded from a CDN or vendored, plus its CSS to override with the direction's tokens, for 35 items that a short vanilla script handles. |
| A server-backed search (HTMX `hx-get` to a suggestion endpoint) | Needs a new route (R-042) and a request per keystroke, for data already in the page. |
| A `<select>` of GPUs on `/` | Changes the control and its steps (R-078, Q-8) and drops free-text search with VRAM hints (R-003). |
| Apply the combobox only below a phone breakpoint | Two behaviours to build, test and keep consistent (R-068) on the same field. The Gate 2 decision says every width. |
| Remove the `<datalist>` from the markup once the combobox exists | Without JavaScript, or if the script fails, the field would lose its suggestions. Keeping it costs 35 `<option>` elements. |

## Consequences

- The combobox is the largest JavaScript addition in the overhaul (about 120 to 150 lines in `app.js`). It needs keyboard and screen-reader checks in QA (R-077) on top of the visual comparison (rule 10).
- With JavaScript off, or if the script fails, the native datalist and the server-side resolution behave exactly as today (R-071, R-078).
- The hidden `gpu_id` contract, the element ids (`gpu_search`, `gpu_id`, `gpu-options`, `clear-gpu`) and the `.chip-example` data attributes stay frozen, because both the old code paths and the combobox depend on them. The mockup's `data-chip` attributes are mockup-only and are not built.
- The change touches `static/js/app.js` and `static/css/styles.css`, so it bumps `ASSET_VERSION` (rule 21, R-079).
- Known issue that stays out of scope (ADR-0008): after htmx restores a page from its history cache (a cache hit), listeners attached to `gpu_search` are lost, as they are today, so the restored page falls back to free text resolved on submit (R-003). A cache miss reloads the page and the combobox starts normally.
