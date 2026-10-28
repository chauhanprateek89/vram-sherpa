# ADR-0012: Landing navigation uses plain GET links and a plain form; `Clear` becomes a link to `/`

Date: 2026-10-28 · Status: proposed · Requirements: R-083, R-001, R-007, R-013, R-039, R-071, R-076, R-077, R-080, R-081 · Related: ADR-0005 (combobox), ADR-0008 (htmx history), ADR-0010 (redirect)

## Context

Every way out of `/` goes to `/results`: submitting the box (Enter or `Check compatibility`), picking a styled suggestion (one tap, no further press, Q-16), and pressing a chip, with and without JavaScript. Every way back goes to `/` with an empty box: `Clear` on `/results` (R-001), the logo and wordmark (R-076) and `Estimator` (R-039). Back from `/results` must show the full landing page (R-083). The box takes focus on load (Q-17).

Today the results form is htmx-driven (`hx-get="/results/partials/content"`, `hx-push-url="true"`), htmx keeps a 10-entry `localStorage` snapshot cache of the `<body>` (ADR-0008), and `Clear` is a `<button type="button" id="clear-gpu">` whose `initGpuInputs` handler empties the field in place. The chips are `.chip-example` buttons that fill the form through `initExampleChips`.

## Decision

1. **The landing form is a plain HTML form** (`method="get" action="/results"`, no `hx-*`, no `hx-boost`). Enter in the box and the submit control both do a normal browser navigation to `/results?gpu_search=<text>`, with or without JavaScript, and the server's rule decides the rest (ADR-0011). The form sends no `context_tokens`, so the results page computes for its default, 2048 tokens, with `2k` selected and no filters (R-006).
2. **Picking a styled suggestion submits.** The box carries `data-submit-on-choose`. In `initGpuCombobox`, after `choose()` writes the exact label and dispatches `input` and `change` (unchanged), it closes the list and, when the input has that attribute, calls `input.form.requestSubmit()` (falling back to `form.submit()` where `requestSubmit` is missing). The URL is `/results?gpu_search=<exact label>`, which the server's exact-match step resolves to that GPU, so the results page shows that label as the hardware and in `Search GPU`. On `/results` the input has no such attribute, so choosing keeps today's behaviour (fill, then wait for `Check compatibility`). Enter with no active option, or with the list closed, submits as today.
3. **Without JavaScript** the box keeps its `list="gpu-options"` and shows the native list (R-001). Picking a native option fills the box; the visitor then presses Enter or the submit control. R-083's "without a further press" criterion is read as applying to the styled list, which needs JavaScript.
4. **Empty submit.** Without JavaScript, `gpu_search=` (or only spaces) reaches `/results`, which redirects to `/` (ADR-0010). With JavaScript, a small submit handler on `#landing-form` cancels the submit when the box is blank after trimming and keeps focus in the box: the address stays `/`, no message shows, and no extra history entry or round trip is made.
5. **Chips are links** (ADR-0009): `<a class="chip" href="/results?gpu_id=<id>&context_tokens=2048">RTX 4060 8GB</a>`, `…?vram_gb=8&context_tokens=2048` and `…?vram_gb=24&context_tokens=2048`, built with `_results_url`. They work without JavaScript, open in a new tab, and need no script. The GPU chip passes `gpu_id`, not the label, because R-007 binds it to a GPU id.
6. **Focus on load** is the `autofocus` attribute on the box, which works without JavaScript. The combobox opens on input, not on focus, so the list stays closed on load (state L1).
7. **`Clear` becomes a link.** On `/results`, `#clear-gpu` becomes `<a id="clear-gpu" class="btn-clear" href="/">Clear</a>`, keeping the id, the label word for word and the secondary-button look (R-001, R-068). `initGpuInputs` drops its `Clear` click handler. It is a normal navigation, so it leaves the htmx-managed page entirely and loads `/` with an empty box, with or without JavaScript.
8. **Logo, wordmark and `Estimator`** already link to `/` in `base.html` and keep doing so. `Estimator` keeps `aria-current="page"` on `/` and `/results` (R-039).
9. **History.** `/` and the first `/results?…` are separate documents reached by normal navigations, so moving between them with Back and Forward is the browser's own document history (back-forward cache or a fresh load), not htmx's. htmx is loaded on `/` (it is in `base.html`) but `/` has no `hx-*` attribute, so htmx pushes nothing there and never snapshots it. Pressing Back from `/results` to `/` therefore shows the full landing page, served fresh or restored by the browser, never a fragment. Inside `/results`, in-place refreshes keep ADR-0008's behaviour unchanged. Because the landing page has no htmx swaps, ADR-0008's known issue (listeners lost after an htmx cache hit) cannot occur on it.
10. **Caching.** `/` is an HTML response, so the ADR-0004 middleware sends `Cache-Control: no-cache` and the `Vary` header, as R-080's check URL 1 requires. The redirects of ADR-0010 set the same headers themselves. Nothing new is cached.

Guards (pytest, rule 19): the landing form has `method="get"`, `action="/results"`, no `hx-` attribute and one text input named `gpu_search` with `autofocus` and `data-submit-on-choose`; the submit control is `type="submit"` with the accessible name `Check compatibility`; each chip is an `<a>` whose `href` is the URL above, and following it returns 200 with the hardware R-007 states; `#clear-gpu` on `/results` is an `<a href="/">` with the text `Clear`; the header's brand and `Estimator` links point to `/`. The `node --check` test covers the script's syntax. Picking, focus, the empty-submit guard and Back are QA's checks in a real browser (rule 10).

## Alternatives considered

| Option | Why not |
| --- | --- |
| `hx-get` the landing form into the page and push `/results?…` | The landing body would land in htmx's snapshot cache, so Back could restore a stale landing page, and the results page would need its form and layout swapped in. A full navigation is simpler and is what "takes you to the page" means. |
| `hx-boost` on the landing page | Same history concerns, and the boosted swap of `<body>` loses the listeners `app.js` attaches on `DOMContentLoaded` (ADR-0008's known issue). |
| Navigate on pick with `location.href = "/results?gpu_id=…"` | Builds a second URL shape in JavaScript. Submitting the form keeps one path, and the exact label is resolved by the same server rule as typed text. |
| Keep `Clear` as a button and navigate from its handler | Needs JavaScript to reach `/`, which R-001 requires with or without it. |
| Chips as buttons that fill the box and submit | Needs JavaScript (R-007 requires chips to work without it) and would route a GPU chip through text matching instead of its id. |
| Focus the box from `app.js` | Works only with JavaScript and runs after the script loads; `autofocus` is earlier and needs nothing. |

## Consequences

- `app.js` changes: submit-on-choose in the combobox, the landing empty-submit guard, removal of the `Clear` handler, and the `data-box-vram` copy-back (ADR-0011). Each task that touches it bumps `ASSET_VERSION` (rule 21).
- `styles.css` needs `a.btn-clear` and `a.chip` to keep the button look (the current rules target the classes, but link defaults such as underline and inline display need resetting).
- A visitor who picks a suggestion gets a URL with the label in `gpu_search`; it is shareable and reloads to the same GPU.
- `UI: yes` (R-083, R-001, R-007). The mockups must keep the box, a real submit control, the chips as links, and `Clear` as a link with the secondary-button look.
