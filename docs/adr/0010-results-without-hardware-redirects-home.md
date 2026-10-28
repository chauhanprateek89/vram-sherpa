# ADR-0010: A full page load of `/results` with no hardware gets a 302 to `/`

Date: 2026-10-28 · Status: proposed · Requirements: R-013, R-005, R-008, R-032, R-080, R-081, R-083 · Follows: ADR-0008 fix 4 (same redirect pattern)

## Context

Today `/results` with no `gpu_id`, `gpu_search` or `vram_gb` renders the full results page at the assumed 8 GB with the default note (R-005). After R-083, the landing page is where a visitor with no hardware belongs, and the accepted shape says "`/results` with no GPU, VRAM or search text also goes to `/`" (Q-18). Three cases must keep today's behaviour:

- htmx requests (`HX-Request: true`): an in-place refresh after the user empties `Search GPU` and `Available VRAM (GB)` cannot follow a redirect into `#results-shell`, and R-005 keeps the 8 GB fragment for it. htmx's history-restore fetch also carries `HX-Request: true` and must get the full page (ADR-0008 fix 3).
- Hardware that is present but unusable (`vram_gb=abc`, `vram_gb=0`, `gpu_id=no_such_gpu`) keeps the approved input-error state at 8 GB (R-008, R-067 state 6).
- The empty landing box submits `gpu_search=` (blank), which must land back on `/` (R-083).

## Decision

1. **Where.** The check is the first statement of the `results` handler, before `_results_page_context`, so a redirect makes no database query. It is not a middleware, so `host_guard` structurally runs first and an unknown `Host` gets the 400 JSON before any redirect (rule 8, rule 18's smoke check).
2. **When.** Redirect when both hold:
   - `request.headers.get("HX-Request") != "true"` (the exact-`true` test of ADR-0008 fix 4; a missing header, `false` or any other value counts as a page load); and
   - none of `gpu_id`, `gpu_search`, `vram_gb` is a string with a non-whitespace character. A small helper (for example `_has_hardware_input(gpu_id, gpu_search, vram_gb)`) states this once.
   Any other parameter (`context_tokens`, `family`, filters) is ignored for the decision and not carried over.
3. **Response.** `RedirectResponse("/", status_code=302, headers={"Cache-Control": "no-cache", "Vary": "HX-Request, HX-History-Restore-Request"})`. The `Location` is the literal path `/`: no scheme, no host, never built from `request.url` or `Host`, so behind the tunnel the browser stays on `https` (rule 8). The status is passed explicitly, because Starlette's default is 307. It is never 301 or 308 (R-013). The response has no `text/html` body, so the ADR-0004 middleware skips it, and it sets both headers itself, as ADR-0008's redirect does.
4. **What is not redirected.**
   - Any request with `HX-Request: true`, including htmx's history-restore fetch, is answered exactly as today (fragment, or full page for a restore).
   - A request with any hardware parameter present, usable or not, is rendered as today (R-005, R-008, Q-18).
   - The partial routes are unchanged. An old `/results/partials/content` link with no hardware, opened as a page, still gets ADR-0008's 302 to `/results?<same query>` (R-081 requires that `Location` byte for byte), and that request then gets this 302 to `/`. Two hops, both relative.
5. **Validation order.** FastAPI validates the declared query parameters before the handler runs, so `/results?context_tokens=abc` with no hardware returns R-008's 422 JSON, not the redirect. This matches ADR-0008 fix 4 and keeps R-008's 422. R-013's "whatever other parameters it carries" read literally would redirect it; the conflict is raised with the user rather than resolved by loosening the parameter types.

Guards (pytest, `follow_redirects=False`, rule 19):

- Each of `/results`, `/results?context_tokens=4096&family=qwen`, `/results?gpu_search=%20%20&vram_gb=` and `/results?gpu_id=&gpu_search=&vram_gb=&context_tokens=2048` returns 302 with `Location: /` exactly, `Cache-Control: no-cache`, both names in `Vary`, and no body. The same with `HX-Request: false`.
- `/results` with `HX-Request: true` returns 200 with the fragment and the default note; with `HX-Request: true` and `HX-History-Restore-Request: true` it returns the full page.
- `/results?vram_gb=abc`, `/results?vram_gb=0` and `/results?gpu_id=no_such_gpu` return 200 with results at 8 GB (and the R-008 error for the first two).
- With `allowed_hosts` not `*`, `/results` with an unknown `Host` returns the 400 JSON with no `Location`.
- `/results/partials/content` without `HX-Request` and with no query still returns 302 `Location: /results` (R-081, unchanged).

## Alternatives considered

| Option | Why not |
| --- | --- |
| Render the landing page at `/results` instead of redirecting | The address would stay `/results`, but R-083 wants `/`, and two URLs would serve the landing page. |
| 303 or 307 | Both are allowed by R-013 and behave like 302 for a GET. 302 matches ADR-0008 and is the most widely understood. |
| 301 or 308 | Browsers cache permanent redirects, which would outlive a rollback and could be replayed for htmx's own request to `/results`. R-013 forbids them. |
| Redirect htmx requests too, with `HX-Redirect` or `HX-Location` | R-013 and R-005 keep the 8 GB fragment for in-place refreshes; a sudden jump to `/` while the user is editing the form on `/results` would lose their filters. |
| A middleware on `/results` | Middleware order decides whether the host check runs first (ADR-0008, alternatives). In the handler the order is structural. |
| Take `context_tokens` and `recommended_only` as strings so the redirect wins over the 422 | Changes R-008's 422 for every `/results` request and moves validation into the handler, for a malformed URL nobody gets from the app. Left to the user (conflict raised). |

## Consequences

- An empty landing submission without JavaScript, a bare `/results` link and a results-page form submitted without JavaScript after both hardware fields were emptied all land on `/` with an empty box.
- R-005's 8 GB default and note now show only for htmx requests and for unusable hardware.
- `test_r078_acceptance.py` reads the default note from a bare `/results` today; it moves to an htmx request or to `/results?gpu_id=no_such_gpu`.
- The redirect has no HTML body, so R-080's check URLs are unaffected (none of them is a bare `/results`).
- `UI: no` on its own: the redirect shows an existing page.
