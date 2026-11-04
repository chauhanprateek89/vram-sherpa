# ADR-0016: The form keeps the view through `htmx:configRequest`, and without JavaScript `Apply filters` carries it through one redirect

Date: 2026-11-02 · Status: proposed · Requirements: R-089, R-013, R-090, R-078 · Decisions: Q-34 (Gate 1, 2026-11-02) · Constitution: rules 8 and 11 · Touches: ADR-0006, ADR-0010

## Context

R-089 splits the form's actions in two. Context presets, filters and `Apply filters` keep `verdict` and `page_size` and go to page 1. `Check compatibility` and Enter return to the gist, with none of the three in the address, with or without JavaScript. R-089 also requires that after Back, a filter change applies to the verdict that view shows.

All of these go through one `<form id="search-form">`, whose single `hx-get` has three triggers: `submit`, `change from:.js-auto-filter` and `keyup changed delay:350ms from:.js-auto-filter-text`.

Hidden inputs cannot do this. They would be sent by every action, including `Check compatibility`. The form sits outside `#results-shell`, so they would also go stale after an in-place swap.

Without JavaScript there is a further limit. A submit button adds exactly one `name=value` pair, and `formaction` cannot help, because a GET form replaces the action's query with the form data. So plain HTML cannot make one button send both `verdict` and `page_size` while another sends neither.

## Decision

1. **The server states the view on the answer tile.** `#summary-banner` gains `data-verdict` when a valid verdict is in the query, including one whose count is 0, and `data-page-size` when it is not 5. Only canonical values are written (ADR-0014). The tile is in every results response, like `data-box-vram` and `data-vram-invalid` before it.
2. **With JavaScript, one listener.** On `htmx:configRequest`, when the event's target is `#search-form`: if `detail.triggeringEvent.type` is `submit`, add nothing. Otherwise copy `data-verdict` and `data-page-size` from the current `#summary-banner` into `detail.parameters`. The form has no `page` field, so the request is page 1. Checked against htmx 1.9.12: `triggeringEvent` is on the request config, and parameters added there go into the URL of a GET.
3. **`HX-Push-Url` needs no change.** It repeats the request's raw query, so the address carries `verdict=tight` and no `page` (R-089).
4. **The DOM is the current view.** After Back or Forward the page has been reloaded from its address (ADR-0017), so `data-verdict` belongs to the entry on screen. After `Fits`, then Back to the gist, `8k` stays on the gist (R-089). A count of 0 keeps `data-verdict`, so a later change can bring the list back (R-089).
5. **Without JavaScript, `Apply filters` names the view in one value.** The button inside `<noscript>` renders as `<button type="submit" name="apply_view" value="<verdict>:<size>">Apply filters</button>` when a verdict is in the query, and as today, with no name, when none is. `Check compatibility` keeps no name and comes first in the form, so it is also the default button for Enter, and sends no view.
6. **`results` turns `apply_view` into the real parameters.** After the no-hardware check and before any database work, a request without `HX-Request: true` that carries `apply_view` gets a `302` to the relative `/results?<q>`, where `<q>` comes from `view_query` with the parsed verdict and size and no page. An invalid value is dropped, which gives the gist. `apply_view` is never rendered, never appears in a link and is never left in an address. The redirect sets `Cache-Control: no-cache` and `Vary` itself, as the other redirects do (ADR-0008, ADR-0010).
7. **Chips, `Try Q4` and `Lower context` stay plain full-page links, with or without JavaScript,** as they are today. They carry the view in their URLs (ADR-0014 point 6). A full load also re-renders the form's checkboxes, which an in-place swap would leave stale (R-031).
8. **`#search-form` gains `autocomplete="off"`,** so a browser restoring form state on reload cannot put values in the form that differ from the address. The GPU box already has it. Every Back and Forward is now a reload (ADR-0017), so this path is no longer rare (R-081, R-088).

## Alternatives considered

| Option | Why not |
| --- | --- |
| Hidden `verdict` and `page_size` inputs in the form | Sent by `Check compatibility` too, and stale after a swap, because the form is outside the swap target. |
| `hx-vals="js:…"` on the form | Runs an expression through `eval` from an attribute, on every trigger, and still needs the trigger type. The listener is plain script in `app.js`. |
| Read the view from `location.search` | Would forward an invalid `verdict=green` from the address into the next request and the next address. The banner holds only canonical values. |
| A second `hx-get` form or separate triggers for submit and change | Splits one form into two request paths, for one parameter difference. |
| `Check compatibility` carries a marker, hidden inputs always sent, and the server strips them | Redirects on the common path, the plain search, instead of the rare one. |
| `Apply filters` as `name="verdict" value="fits"` alone | Loses `page_size=10`, against R-089. |
| A `<noscript>` hidden `page_size` | Sent by `Check compatibility` too, so the address keeps `page_size=10` on the gist, against R-089. |

## Consequences

- One new query parameter, `apply_view`, exists only for the length of a redirect, and only without JavaScript. It is not a stored value (rule 11), adds no route (R-042), and its `Location` is a relative path (rule 8).
- A no-JavaScript `Apply filters` with a verdict selected takes one extra hop, which the visitor does not see.
- If unpkg is unreachable and htmx does not load while JavaScript runs, `<noscript>` does not apply. The links still work as full pages and `Check compatibility` still submits, so a filter change needs that button. That limit exists today.
- `UI: no`: nothing here draws anything. The `<noscript>` button keeps its label and style.
