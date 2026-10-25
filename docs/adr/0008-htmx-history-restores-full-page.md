# ADR-0008: htmx history entries point at full-page URLs, and a cache miss reloads the page

Date: 2026-10-25 · Status: proposed · Requirements: R-081, R-032, R-078, R-080 · Amends: ADR-0004 (`Vary` value)

## Context

At Gate 2 round 1 the user decided to fix this bug in the overhaul: after a back navigation, the page can be replaced by the bare results fragment (R-081). The templates load htmx **1.9.12** from unpkg (`base.html`). The analysis below was checked against that version's `dist/htmx.js`.

- The form in `index.html` has `hx-get="/results/partials/content"`, `hx-target="#results-shell"` and `hx-push-url="true"`. With `"true"`, htmx pushes the response URL (`determineHistoryUpdates`: `responsePath || requestPath`). So the address bar and the history entry become **`/results/partials/content?…`**, not `/results?…`. ADR-0004's note said `/results?…`. That was wrong.
- Before each push, htmx saves the current `<body>` HTML to `localStorage` (`htmx-history-cache`, `historyCacheSize` 10). On popstate, `restoreHistory` swaps the cached body back in. On a miss (the entry fell out of the 10-entry cache, the storage was cleared, or the storage is unavailable), `loadHistoryFromServer` runs a GET on the entry's URL with `HX-Request: true` and `HX-History-Restore-Request: true`, and swaps the response into `<body>`.
- `results_partial_content` always returns `results_content.html`, and `results` returns it for any `HX-Request: true`. Either way the miss swaps a fragment into `<body>`, and the header, form and footer disappear.
- The same pushed URL also breaks reload and sharing: opening `/results/partials/content?…` directly returns the bare fragment, with no stylesheet. This contradicts R-032 ("a URL that reproduces the same results when opened directly"), and product raised it as Q-13.
- htmx 1.9.12 takes an `HX-Push-Url` response header in preference to the `hx-push-url` attribute. It has `htmx.config.refreshOnHistoryMiss` (default `false`), which makes a miss call `window.location.reload()` instead of an XHR. It reads config from `<meta name="htmx-config">` when it initialises.

## Decision

Three small changes, all inside the frozen interface of R-078 and ADR-0006: no route, parameter, `hx-*` attribute or target changes.

1. **Push the full-page URL.** `results_partial_content` sets the response header `HX-Push-Url: /results?<query>`, where `<query>` is the request's raw query string unchanged (`request.url.query`, already percent-encoded, with repeated `family` and `quant_bucket` kept in order). If the query is empty, the header is just `/results`. The form keeps `hx-push-url="true"`, and the header overrides it. Every history entry, the address bar and any copied link now name `/results?…`, which `results` renders as the full page for a normal request.
2. **Reload on a miss.** `base.html` gains `<meta name="htmx-config" content='{"refreshOnHistoryMiss":true}'>` in `<head>`, before the htmx script. A miss becomes a real browser load of `/results?…`, sent without `HX-Request`. The browser gets the whole document, the head scripts run, and `app.js` initialises on `DOMContentLoaded`, so the restored page works like a fresh one. A cache hit behaves as it does today.
3. **Server safety net.** `results` returns the fragment only when `HX-Request` is `true` **and** `HX-History-Restore-Request` is not `true`. Otherwise it returns `index.html`. This is the pattern the htmx documentation recommends for servers that branch on `HX-Request`. It covers the case where the meta tag is lost in a template rewrite. htmx 1.9.12 extracts `<body>` from a full-document response before swapping, so this path cannot nest a second header or footer. Because `/results` now varies on that header too, the ADR-0004 middleware sets `Vary: HX-Request, HX-History-Restore-Request` on HTML responses, in place of `Vary: HX-Request`.

Guards (pytest, rule 19):

- `GET /results/partials/content?vram_gb=24&context_tokens=4096&family=a&family=b` with `HX-Request: true` carries `HX-Push-Url: /results?vram_gb=24&context_tokens=4096&family=a&family=b`.
- `GET /results?…` with both `HX-Request: true` and `HX-History-Restore-Request: true` returns the full page (the header, form and footer markers). With only `HX-Request: true`, it still returns the fragment (R-032).
- The rendered `base.html` contains the `htmx-config` meta tag with `refreshOnHistoryMiss` set to `true`, before the htmx `<script>`.
- HTML responses carry both names in `Vary`.

R-081's browser steps (forcing a miss by clearing `htmx-history-cache`, and making 12 refreshes against the 10-entry cache) are QA's check in a real browser (rule 10).

## Alternatives considered

| Option | Why not |
| --- | --- |
| Only the server branch (`HX-History-Restore-Request` returns the full page), with the URL left at `/results/partials/content` | Means changing the partial endpoint as well, still leaves reload and shared links broken, and `loadHistoryFromServer` swaps a new `<body>` without rerunning `app.js`'s `DOMContentLoaded` setup, so the theme toggle and GPU input wiring would be dead on the restored page. |
| Only `refreshOnHistoryMiss`, with the URL unchanged | The reload fetches `/results/partials/content?…` without `HX-Request`, and that endpoint returns the fragment anyway. The bug moves rather than going away. |
| Change the form to `hx-get="/results"` | Fixes the pushed URL with no header, but changes an `hx-*` attribute that ADR-0006 and R-078 freeze, and leaves `/results/partials/content` with no caller. The response header gets the same effect with no template change. |
| `historyCacheSize: 0` or `hx-history="false"` (always fetch from the server) | Turns every Back into a server round trip, and still needs fix 1 or fix 3 to avoid fragments. |
| Make the partial endpoints return the full page to non-HTMX requests | Changes the as-built behaviour of the three fragment endpoints (R-032, R-078). Fix 1 makes it unnecessary for new history entries. See consequences for old links. |
| Upgrade htmx to 2.x | Out of scope for a restyle, changes extension and attribute behaviour, and 2.x has the same miss path. Not needed for this fix. |

## Consequences

- A Back or Forward that misses the cache always shows the full page, with working scripts (R-081). A cache hit is unchanged.
- As a side effect, reload and copy-and-paste of the address bar after an in-place refresh show the full page (Q-13). Fixing back navigation requires this: the history entry and the address bar are the same URL.
- Links and history entries created before the release still name `/results/partials/content?…`, and opening one shows the fragment as it does today. Fixing those would mean fix 5 in the alternatives, which is a separate decision for the user.
- A miss now costs one full page load instead of one fragment request. With a 10-entry cache this is rare.
- Pre-existing and not changed by this ADR: on a cache **hit**, htmx swaps `<body>` from `localStorage`, so listeners that `app.js` attached to individual elements (`theme-toggle`, `gpu_search`, `vram_gb`, `clear-gpu`, `.chip-example`) are lost on the restored page. Event delegation on `document` survives, which is why the quantisation selector keeps working. Rerunning the init functions on `htmx:historyRestore` would fix it. It is flagged for the user and is not part of R-081.
- If htmx is ever upgraded or its pin changes, the three guards above must be rechecked against the new version's history code.
- `UI: no` (R-081). No visual change and no mockup.
