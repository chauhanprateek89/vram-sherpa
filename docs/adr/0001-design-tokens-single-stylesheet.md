# ADR-0001: One stylesheet built on design tokens, with shared markup for shared elements

Date: 2026-10-25 · Status: proposed · Requirements: R-067, R-068, R-070, R-072, R-077, R-079

## Context

The overhaul restyles every screen in one visual direction that the user picks at Gate 2 from 2 or 3 candidates (R-068). Today `static/css/styles.css` is one 1233-line file. It already uses CSS custom properties for colour and fonts, but spacing, type sizes and radii are hard-coded per rule. The dark palette is written out twice (once inside `@media (prefers-color-scheme: dark)` for visitors with no saved choice, and once under `:root[data-theme="dark"]` for the toggle, R-040), and the two copies can drift. The same kind of element is styled by different classes on different pages: the verdict is `.state` on cards but `.badge` on model detail, and the gated marker is `.tag` on cards but `.badge-gated` on model detail. R-068 requires identical computed styles for the same element on every screen.

There is no CSS build step, and CI and the Dockerfile support none (`pip install .` copies files; nothing compiles them). Rule 21 makes `ASSET_VERSION` the only cache-busting for assets.

## Decision

Keep a single hand-written `src/vramsherpa/static/css/styles.css`, rewritten for the overhaul in this fixed section order: tokens, reset and base, layout, components, page-specific rules, utilities. All visual values come from custom properties on `:root`: semantic colour tokens (surface, text, muted text, border, accent, focus, and per-verdict foreground, background and bar colours, reserve band), a type scale, a spacing scale, radii, shadows and motion durations. Components use only semantic tokens, never raw values, so the chosen Gate 2 direction becomes a set of token values plus component rules. Theming keeps today's mechanism, which works without JavaScript: light tokens on `:root`, dark tokens under both `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) }` and `:root[data-theme="dark"]`, with a pytest that fails if the two dark blocks declare different properties or values. The stylesheet is mobile-first: base rules target 390 px, and `min-width` media queries widen the layout (at most two breakpoints, recorded as a comment beside the tokens because custom properties cannot be used in media queries). Each element kind in R-068 gets exactly one component class (one button family with primary and secondary modifiers, one field style, one chip, one disclosure, one verdict label, one gated marker), and the verdict label and gated marker are rendered by Jinja macros in a new `templates/components.html`, imported by `results_list.html`, `results_summary.html` and `model_detail.html`, so both pages emit the same markup. Every task that changes the stylesheet or `app.js` bumps `ASSET_VERSION` (rule 21, R-079).

## Alternatives considered

| Option | Why not |
| --- | --- |
| Split into several CSS files (tokens, components, pages) linked separately | More requests and more `<link>` tags to keep versioned, for no gain at this size. `@import` would load the files one after another. Without a build step there is nothing to bundle them. |
| CSS `light-dark()` with `color-scheme`, so each colour token is written once | Removes the duplicated dark block, but Safari before 17.5 treats the whole declaration as invalid, so those visitors would lose every colour rather than fall back to one theme. The duplicated block plus a test gives the same safety without that risk. |
| A CSS framework or utility library (Tailwind, Pico, Open Props) | Tailwind needs a build step that CI and the Dockerfile lack. A classless or token library would impose its own look on top of the chosen direction and add a dependency for what is about 1500 lines of CSS. |
| Sass or PostCSS for variables and nesting | A build step and a Node toolchain in CI and the image. Native custom properties and nesting already cover what we need. |
| Keep per-page classes and align their values | Nothing would stop the two pages drifting again, and R-068 asks for identical computed styles. One class and one macro per element kind enforces that by construction. |

## Consequences

- Switching between the Gate 2 directions, or retuning one later, mostly means editing token values.
- The rewrite replaces most of `styles.css`, so overhaul tasks that touch it will conflict with one another. The plan should order them (tokens and base first) rather than run them in parallel, and each one bumps `ASSET_VERSION`.
- The dark palette is still written twice. The new pytest is what keeps the copies equal.
- `templates/components.html` matches the existing `templates/*.html` package-data glob, so it ships in the image without a `pyproject.toml` change.
- Contrast (R-077) can be checked once per token pair rather than per rule. QA still measures it in a real browser (rule 10).
- `results_table.html` is rendered by no route and is not restyled.
