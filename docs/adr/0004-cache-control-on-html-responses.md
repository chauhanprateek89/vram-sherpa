# ADR-0004: Middleware sends `Cache-Control: no-cache` on every HTML response

Date: 2026-10-25 · Status: proposed · Requirements: R-080, R-079, R-041, R-032, R-042, R-043

## Context

Constitution rule 9 says HTML responses send `Cache-Control`, so that a deploy shows on the next page load. No route or middleware sets it today. Without it, browsers may reuse a cached page heuristically, and a redesign is the release where that hurts most: stale HTML still references `styles.css?v=20261024a`, while the new HTML references the new version (rule 21, R-079). At Gate 1 (Q-9) the user decided the overhaul ships with the header, recorded as R-080.

HTML comes from eight routes in `src/vramsherpa/main.py`: four pages, the three `/results/partials/*` fragments and `/results/why/{variant_id}`. `/results` returns either a full page or a fragment depending on the `HX-Request` header (R-032), so one URL has two representations. `/healthz` and the error responses are JSON. `/static` is served by Starlette's `StaticFiles`, which sends `ETag` and `Last-Modified` but no `Cache-Control`. The existing `host_guard` is an `app.middleware("http")` function. Cloudflare does not cache HTML by default, and there is no other shared cache in front of the app.

## Decision

Add a second HTTP middleware in `main.py`, registered in `create_app` next to `host_guard`. For any response whose `Content-Type` starts with `text/html` and that does not already set the header, it sets `Cache-Control: no-cache` and appends `HX-Request` and `HX-History-Restore-Request` to `Vary` (the second name was added by ADR-0008, 2026-10-25). It keys on content type rather than a list of paths, so any future HTML route is covered automatically, and it leaves JSON (`/healthz`, 400, 404, 422) and `/static` untouched. `no-cache` lets a browser store the page but forces a revalidation on every normal load. The app sends no validators, so in practice each load fetches fresh HTML, which in turn names the current `ASSET_VERSION`. Asset caching does not change: CSS and JS stay cache-busted only by `?v=ASSET_VERSION` (rule 21), and font files by versioned names (ADR-0002). Tests check the header on every HTML route and fragment, and check that `/healthz` and `/static` do not get it. The release smoke test gains a check that `GET /` carries `Cache-Control` (an addition rule 18 allows).

## Alternatives considered

| Option | Why not |
| --- | --- |
| `no-store` | Also meets rule 9, but it stops browsers keeping the page in the back/forward cache, so Back becomes a full reload. Nothing on these pages is private. |
| `max-age=60` or similar | A deploy could stay invisible for up to that long, which goes against the wording of rule 9. |
| Set the header in each route, or in a `render()` helper | Eight call sites to remember, and a new route can forget it. The middleware needs one place and one test. |
| A path list (`/`, `/results*`, `/models/*`, `/how-it-works`) in the middleware | Has to be kept in step with the routes. Keying on content type cannot drift. |
| Long-lived `immutable` caching for `/static` at the same time | A separate performance decision. It is safe for CSS and JS under rule 21, but fonts and images rely on discipline about versioned names, and that should be proven first. |
| `ETag` on HTML for cheap 304s | Means hashing every rendered body. The pages are small, and there is no shared cache to save. |

## Consequences

- After a deploy, the next normal page load gets the new HTML and therefore the new CSS and JS. Back and forward navigation may still show the bfcache copy of a page that is already open, which is acceptable.
- `Vary: HX-Request, HX-History-Restore-Request` stops a browser reusing the fragment form of `/results` for a full page load, or the reverse.
- This adds a tiny per-request cost on top of `host_guard`'s.
- Found while designing: the htmx history-miss bug. At Gate 2 round 1 the user brought it into scope (R-081). The mechanism is decided in [ADR-0008](0008-htmx-history-restores-full-page.md). The pushed URL is in fact `/results/partials/content?…`, not `/results?…` as this note first said.
