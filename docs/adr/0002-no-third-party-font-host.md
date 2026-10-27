# ADR-0002: Fonts come from the system stack or are self-hosted, never from a third-party host

Date: 2026-10-25 · Status: proposed · Requirements: R-068, R-069, R-070, R-077

## Context

`base.html` loads IBM Plex Sans and IBM Plex Mono from Google Fonts: a render-blocking cross-origin stylesheet plus two preconnects. So every visitor's browser contacts Google before the page can paint text in the brand font, and Google receives the visitor's IP address. The app is public, free and read-only, and promises to store nothing a visitor enters (rule 11). Its served code depends on nothing but Postgres (rule 15), yet its first paint depends on a third party. The overhaul gives the app a new visual identity (R-068), and the uiux agent is now producing 2 or 3 directions that will each name typefaces, so the delivery rule has to be settled before the directions are finalised.

There is no build step. Static files ship through `[tool.setuptools.package-data]` in `pyproject.toml`, which today matches only `static/css/*.css` and `static/js/*.js`. The Dockerfile runs `pip install .` (not editable), so a file that no glob matches is missing from the image even though tests, which import from `src/`, still pass. `ASSET_VERSION` reaches only URLs written in templates, not `url()` references inside the stylesheet.

## Decision

The app loads no fonts from a third party. Each Gate 2 direction uses one of two options. (a) A system font stack (`system-ui` and friends for text, `ui-monospace` for figures), which costs no bytes. (b) At most two self-hosted families under the SIL Open Font License or a similarly redistributable licence, as WOFF2 files subset to Latin, at most four files and 200 KB in total. Option (b) means: the files live in `src/vramsherpa/static/fonts/` with their licence text; `@font-face` rules in `styles.css` use `font-display: swap`; the primary text face is preloaded from `base.html`; and the file names carry a version (for example `display-v1-600.woff2`), because `url()` inside the stylesheet does not get the `?v=` query. `pyproject.toml` package-data gains `static/fonts/*`. The Google Fonts `<link>` and both preconnects are removed whichever option is chosen.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Keep Google Fonts (or another font CDN) | A cross-origin, render-blocking request on every first visit. It sends visitor IPs to a third party, which German courts have ruled a GDPR breach for Google Fonts (LG München, 2022). It breaks the look when the CDN is blocked or slow, and it adds an outside dependency to a site that otherwise needs none. |
| Self-host with no limit on families, weights or size | Page weight and layout shift on a phone connection count against R-069's first-screen goals. Two families and four files cover a text face plus a figure or display face. |
| Variable fonts with no subsetting | Full variable files often exceed 300 KB each. Subset static weights are smaller for the 2 to 4 weights a direction actually uses. |
| Inline the fonts as base64 in the CSS | Inflates the stylesheet by about a third more than the font itself, and every CSS-only change makes visitors re-download the fonts. |

## Consequences

- No visitor data goes to a font host, and pages render with the right fonts when the visitor is offline from Google or behind a blocker.
- The uiux agent must pick freely licensed typefaces that are available as files (Google Fonts' catalogue is mostly OFL and can be downloaded and self-hosted). Commercial fonts are excluded by rule 2.
- Font binaries are committed to a public repo, together with their licence file.
- The package-data glob is a trap: if `static/fonts/*` is missing, local tests pass but the production image serves 404 for the fonts. The plan adds a pytest that every file under `static/` and `templates/` matches a package-data glob, and adds to the release smoke test a check that every `/static/` URL in the home page returns 200 (an addition rule 18 allows).
- Rule 21 was amended at Gate 2 (2026-10-25): any change under `static/`, fonts included, bumps `ASSET_VERSION`. Font files keep their versioned names too, because `styles.css` loads them by a plain `url()` that `?v=ASSET_VERSION` does not reach.
- HTMX 1.9.12 still loads from unpkg.com in the browser. Vendoring it is the same kind of decision but is not needed for the overhaul, so it is not taken here.
