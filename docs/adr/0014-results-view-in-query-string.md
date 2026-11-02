# ADR-0014: The results view lives in the query string, read by one lenient parser, and links are built from the raw query

Date: 2026-11-02 · Status: proposed · Requirements: R-084, R-085, R-087, R-088, R-090, R-032, R-067 · Decisions: Q-33, Q-35, Q-36 (Gate 1, 2026-11-02) · Constitution: rules 8 and 11

## Context

R-088 adds three parameters, `verdict`, `page` and `page_size`, to `/results` and the three `/results/partials/*` routes. States G1 to G13 of R-067 all depend on them. Three things in the code shape how they are carried:

- Each of the four handlers declares its parameters through `Query(...)` and passes them to `_results_page_context`. A declared `page: int` would answer `page=abc` with FastAPI's 422, and R-088 requires 200 with page 1.
- `index.html` includes `results_content.html`, and every route builds its context with `_results_page_context`. Full page and fragments share one code path today, and that is worth keeping.
- `_query_items` rebuilds a query from *parsed* values, so it drops what did not parse. `/results?vram_gb=abc` becomes `context_tokens=2048`, which has no hardware, and following a link built from it would redirect to `/`. R-088 forbids that, and it also requires every other parameter to stay unchanged and in order.

## Decision

1. **One pure module, `src/vramsherpa/results_view.py`,** with no database, no request object and no import from `main.py`. It holds:
   - `ResultsView(verdict, page, page_size)` and `parse_results_view(params)`;
   - `paginate(total, view)`, which clamps the page and gives first, last, total and the last page;
   - `page_window(page, last_page)`, which returns the numbers and gaps of R-087 (`1 … 5 6 7 … 15`);
   - `view_query(raw_query, verdict=, page=, page_size=)`, which builds a link's query.
2. **The handlers do not declare the three parameters.** `_results_page_context` reads them from `request.query_params`, which it already has, as raw strings. No value can produce a 422, no handler signature changes, and all four routes get the same behaviour from one line.
3. **Parsing never fails and never echoes.**
   - `verdict`: the last value if repeated (R-088), matched exactly against `fits`, `tight`, `wont-fit`; anything else, including `FITS` and empty, is no verdict.
   - `page_size`: the last value, exactly `5` or `10`; anything else is 5.
   - `page`: the last value; `re.fullmatch("[0-9]+")`, so `abc`, `-1`, `1.5`, `+2`, `1_000` and non-ASCII digits are page 1, and `0` is page 1. A value too long to fit a 64-bit integer counts as above the last page, so the 4300-digit limit of Python's `int()` is never reached.
   - With no valid verdict, `page` and `page_size` are ignored (R-088).
   - Only canonical values reach a template or a link, so `verdict=%3Cscript%3E` cannot appear in the body.
4. **Clamp, do not redirect.** `paginate` lowers a page above the last page to the last page, in the response only. The address keeps what was asked for until the user next moves (R-088).
5. **Links come from the raw query.** `view_query` splits `request.url.query` on `&`, removes the parts whose decoded key is `verdict`, `page`, `page_size` or `apply_view` (ADR-0016), keeps every other part byte for byte and in order, and appends the new values, leaving out `page` when it is 1 and `page_size` when it is 5. Each control gets `href="/results?<q>"` and `hx-get="/results/partials/content?<q>"` with the same `<q>` (ADR-0018). The paths are literal, so a link cannot leave the site (rule 8). Jinja's autoescaping covers the attribute.
6. **`_query_items` stays for the links that are canonical today:** active filter chips (R-031), `Try Q4` and `Lower context` (R-025). They gain the valid `verdict` and a non-default `page_size`, and no `page` (R-089).
7. **The full page and the partials agree by construction.** Every route calls `_results_page_context`, which computes the view, the counts and the page. `index.html` keeps including `results_content.html`, and each partial renders one of the same includes. The only differences are the shell around them and the `HX-Push-Url` header. A pytest guards this: for G1 to G8, the exact body of `/results/partials/content?<q>` with `HX-Request: true` appears inside the body of `/results?<q>`.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Declare `page: int = Query(1, ge=1)` and `page_size: Literal[5, 10]` | FastAPI answers bad values with 422. R-088 requires 200. |
| Declare them as `str | None` on four handlers | Twelve more parameters to keep in step, passed through one more signature, for no behaviour. Repeated `verdict` also needs `getlist`, or relies on Starlette's undocumented last-wins for a scalar. |
| A `Depends(ResultsView)` | The same result as reading `request.query_params`, with more wiring. `_results_page_context` already has the request. |
| Build links with `_query_items` plus the view | Drops unusable hardware, so `/results?vram_gb=abc` would follow a link to `/` (R-088 forbids it). It also reorders parameters. |
| Redirect a page past the end to the last page | R-088 says no redirect. A stale shared link should simply open. |
| Sessions or cookies for the view | Rule 11 and R-088. |

## Consequences

- No new route (R-084, R-042). `/results/partials/content`'s `HX-Push-Url` and old-link redirect already pass the raw query, so they carry the view with no change (R-081).
- The three parameters do not appear in FastAPI's generated OpenAPI schema. Nothing in the app reads that schema.
- `UI: no` for the mechanism. The states it serves are UI (R-067).
