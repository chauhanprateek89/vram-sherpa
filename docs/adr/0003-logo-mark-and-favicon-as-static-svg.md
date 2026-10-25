# ADR-0003: Inline SVG logo mark, and favicons served from /static

Date: 2026-10-25 · Status: proposed · Requirements: R-068, R-076, R-042, R-070

## Context

At Gate 1 the user accepted that each visual direction includes a logo mark and a favicon (Q-7). Today the header shows a text wordmark, and there is no favicon, so browsers request `/favicon.ico` and get a 404 JSON response. The mark has to work in both themes and at phone width, and it sits in the header on every page (R-076). R-042 lists the app's only routes, so a dedicated `/favicon.ico` or `/manifest.json` route would change that requirement. There is no image pipeline or build step. Static files reach the image only through `pyproject.toml` package-data globs (see ADR-0002).

## Decision

The header mark is an inline `<svg>` stored in a new `templates/logo_mark.html` and included from `base.html` inside the existing `VRAM Sherpa` link, so the mark and the wordmark both load `/` (R-076). Its colours use `currentColor` or the theme tokens, so it follows the light and dark themes with no extra files or requests. It is decorative (`aria-hidden="true"`), because the link text `VRAM Sherpa` already names the link (R-039). Favicons are static files in `src/vramsherpa/static/img/`: `favicon.svg` (which may carry its own `prefers-color-scheme` style), `favicon-32.png`, and a 180 × 180 `apple-touch-icon.png`. `base.html` links them with `?v={{ asset_version }}`, like the CSS and JS. The PNGs are exported once from the SVG on a developer machine and committed. `pyproject.toml` package-data gains `static/img/*`. No route is added, and `/favicon.ico` keeps returning 404 for the few clients that ignore `<link rel="icon">`.

## Alternatives considered

| Option | Why not |
| --- | --- |
| A `/favicon.ico` route or a root-level static mount | Adds a route that R-042 does not list, to serve clients that already get a harmless 404. |
| The mark as an `<img src="/static/img/logo.svg">` | One more request, and an external SVG cannot inherit `currentColor` or the theme tokens, so it would need a second file for dark mode. |
| A PWA manifest (`site.webmanifest`) with maskable icons | Not required. It adds a file, metadata to maintain, and an "install app" promise for a site that has no offline mode. |
| Generating PNGs in CI or the Dockerfile | A build step for three files that change perhaps once. |
| SVG favicon only | Safari before version 26 and some link-preview crawlers ignore SVG icons, and iOS home-screen icons must be PNG. |

## Consequences

- The mark costs no request and cannot fall out of step with the theme.
- Three binary or SVG files are committed. Changing the mark means re-exporting the PNGs by hand.
- Rule 21 bumps `ASSET_VERSION` only for `static/css` and `static/js`. When a favicon changes on its own, the task must still bump `ASSET_VERSION`, or browsers keep the old icon. The guard test and smoke check from ADR-0002 also cover `static/img/`.
- Each Gate 2 direction must supply the mark as one SVG that reads at 16 px (favicon) and at header size.
