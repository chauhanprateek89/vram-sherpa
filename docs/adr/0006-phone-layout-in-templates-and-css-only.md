# ADR-0006: The 390 px layout is built in templates and CSS only, with routes, fields and figures frozen

Date: 2026-10-25 · Status: proposed · Requirements: R-069, R-070, R-071, R-072, R-073, R-074, R-075, R-076, R-078

## Context

The phone layout breaks today. Filters sit open between the form and the results. The answer comes after Top Picks and the filter chips. Model detail is a seven-column table in a horizontal scroll region, and formulas and long names can overflow. The overhaul must fix this at 390 px and 1280 px (Q-5) without changing any route, query parameter, fragment endpoint, figure or verdict (R-078), and it must keep working without JavaScript. Several requirements look like they need code: filters collapsed on phones with an active count (R-069, Q-3), error and GPU messages shown once and refreshed with the results (R-071), a table with no horizontal scroll (R-074), and numeric keyboards (R-070). The form is not re-rendered on an in-place refresh: HTMX swaps only `#results-shell`.

## Decision

All layout work happens in Jinja templates, `styles.css` and small additions to `app.js`. `main.py` keeps every route, parameter and figure, and gains at most new template context values. Specifically:

1. **Frozen interface.** Route paths, query parameter names, form field `name`s, the HTMX attributes and targets (`#results-shell`, `hx-push-url`, `/results/why/…` with `hx-trigger="click once"`), and the ids and data attributes that `app.js` reads (`gpu_search`, `gpu_id`, `gpu-options`, `clear-gpu`, `vram_gb`, `theme-toggle`, `data-model-card`, `data-quant-btn`, `data-variant-pane`, `.chip-example`) do not change. Templates may be reordered and split into new includes (for example `results_top_picks.html`). `results_content.html` still composes the summary and the list, and the three `/results/partials/*` endpoints still return their templates.
2. **Answer first (R-069).** The result headline (the `N models fit or are tight` sentence and counts) moves to the top of the results region. Top Picks go wherever the user picks at Gate 2 (Q-1), as an include.
3. **Messages inside the swapped region (R-071).** The `role="alert"` error list and the `role="status"` GPU feedback are rendered only inside `#results-shell` (in the summary template) and removed from the form in `index.html`. Each response then shows exactly the current query's messages, once. Model detail keeps its own single error list.
4. **Filters disclosure (R-069, R-078).** The filters stay in the one `<form>`, inside a native `<details>` rendered closed, whose `<summary>` reads `Filters (N)`. N is computed on the server from the active filters, the same set as the chips (R-031). An inline script directly after the element opens it before first paint when `matchMedia` reports desktop width. So phones always start collapsed, desktop starts open when JavaScript runs, and without JavaScript the user can still open it. `app.js` updates N when a filter input changes, counting only values the server would accept.
5. **Mobile-first CSS (R-070).** Base rules target 390 px and `min-width` queries add columns. `min-width: 0` on grid and flex children, `overflow-wrap: anywhere` on model names, GPU labels, source URLs and `code`, and no `white-space: nowrap` on text. Fields use 16 px text at phone width, and tap targets are at least 44 px. The header keeps its three items in one row, wrapping if needed. If they do not fit, a `<details>` menu is used, never a JavaScript-only menu.
6. **Numeric keyboards (R-070).** Number fields stay `type="number"` (so `min` and `step` are unchanged) and gain `inputmode="decimal"`.
7. **Model detail variants (R-074).** The `<table>` stays as the desktop markup. At phone width, CSS turns each row into a stacked block, with labels from `data-label` attributes on the cells. Explicit `role="table"`, `row`, `columnheader` and `cell` attributes keep the table semantics that some browsers drop when `display` changes. Provenance stays the last cell. `.table-scroll` is removed.
8. **Breakdown and how-it-works (R-073, R-075).** The breakdown becomes a definition list that wraps inside the card. Formulas wrap rather than scroll. The breakdown's placeholder becomes a distinct loading state, using htmx's `htmx-request` class on the `<summary>`, so no extra JavaScript is needed.

Layout is verified by QA in a real browser at 390 and 1280 px in both themes (rule 10). Pytest checks the markup invariants: frozen ids and attributes, each message rendered once, `inputmode`, `data-label`, and the R-078 reference queries returning the same models, order and figures.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Render the filters twice (a phone drawer and a desktop sidebar) | Duplicate field names submit twice and break the URL contract of R-032. |
| Detect phones on the server (User-Agent) to decide `open` | Unreliable, and it varies the HTML by a header the cache layer does not key on (ADR-0004). |
| CSS-only disclosure (checkbox hack, `::details-content`) | The checkbox hack is poor for accessibility. `::details-content` styling is not yet in every target browser. |
| Server-rendered `open`, closed by script on phones | Phones without JavaScript would start with the filters open, against R-069. |
| An out-of-band HTMX swap (`hx-swap-oob`) to refresh the in-form messages and filter count | Puts the same element in two templates and makes the full-page render emit stray oob markup. Moving the messages into the swapped region is simpler. |
| Replace the model detail table with cards at every width | A table is the right structure for comparing four variants side by side on desktop. Reflowing at phone width keeps both. |
| A JavaScript layout or component framework | No build step, and R-078 requires everything to work without JavaScript. |
| Automated visual regression in CI (Playwright screenshots) | New dev dependencies, browser downloads in CI and brittle baselines. Rule 10 already assigns this check to QA in a real browser. |

## Consequences

- `main.py` changes stay small, mainly the filter count and any wording context, so the R-078 figure checks are low risk.
- Everything that appears on scroll is pure CSS, so intermediate widths (for example 768 px) are not mocked or QA'd. The only guarantee there is no horizontal scroll.
- The inline disclosure script joins the existing inline theme script. Both must survive if a Content-Security-Policy is ever added (none exists today).
- Desktop visitors without JavaScript see the filters collapsed, with the count, and must open them.
- Moving the messages out of the form changes where they sit on screen. The Gate 2 mockups must place them in the results region.
