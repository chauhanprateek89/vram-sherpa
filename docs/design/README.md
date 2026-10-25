# UI/UX overhaul: design (Gate 2, round 1 of 2)

Direction boards for requirements R-067 to R-079. The user picks one direction; round 2 then mocks up every screen and state of R-067 in it.

Gallery (private Artifact, before screenshots and all three directions side by side): https://claude.ai/artifact/GpFFFNrx4duQTg63MYdQQQ
Local copy: `docs/design/gallery.html` (open it from this folder so the relative images resolve).

## Index

| Direction | Mockup (responsive, `?theme=dark` for dark) | Brand sheet (logo, favicon, tokens, contrast, type) |
|---|---|---|
| A. Basecamp: calm field guide, serif answer, ruled rows | `mockups/direction-a-home.html` | `mockups/direction-a-brand.html`, `direction-a-brand-1280.png` |
| B. Altimeter: instrument panel, dark first, segmented gauges | `mockups/direction-b-home.html` | `mockups/direction-b-brand.html`, `direction-b-brand-1280.png` |
| C. Expedition: bold colour blocks, three-step form | `mockups/direction-c-home.html` | `mockups/direction-c-brand.html`, `direction-c-brand-1280.png` |

Renders, per direction `x` in a, b, c: `direction-x-home-390-light.png`, `direction-x-home-390-dark.png`, `direction-x-home-1280-light.png`, `direction-x-home-1280-dark.png` (full page).
Before (today's app, `/results?vram_gb=24&context_tokens=4096`): `before/results-{390,1280}-{light,dark}.png`.
Fonts (all SIL OFL 1.1, woff2, self-hostable): `fonts/` (Young Serif and Hanken Grotesk for A, Saira for B, Archivo for C), licences alongside.

## Measured on the boards (24 GB, 4k)

| | First card starts, 1280 px (target 800) | First card starts, 390 px (target 1688) | Top Picks (Q-1) |
|---|---|---|---|
| Today | 1147 | 2800 (page is 540 px wide in a 390 px window) | above the cards |
| A | 752 | 1409 | above the cards |
| B | 642 | 1002 | side rail on desktop, collapsed on phone |
| C | 668 | 1234 | band after the first cards |

All three: no horizontal scroll at 390, every control at least 44 px tall at 390, no text under 14 px, text 4.5:1 and non-text 3:1 in both themes (computed from the tokens; see the brand sheets).

## Requirement coverage

R-068 identity, logo and favicon (Q-7), R-069 answer first and collapsed phone filters, R-070 phone layout, R-072 card, switcher and gauge, R-076 header and footer, R-077 contrast and focus, and Q-1 Top Picks placement. Round 2 covers the rest of R-067 (states, breakdown, model detail, how it works, errors, empty and unseeded).

## Open points for the build

- Top Picks above the cards (A) only meets the 800 px target with a one-row hero; any error banner or filter chips pushes the first card past it.
- C colours the verdict pane with CSS `:has()`; the build should add a server-side class instead.
- C's gauge gets its 3:1 from an ink outline, not the pastel fill.
- Desktop filter checkboxes in A and C are 36 to 38 px tall; the 44 px rule is phone-only.
- All three drop the second "How it works" link from the form and reword the help text, the "Only Fits and Tight models are listed" note and the Top Picks descriptions (Q-2). The user approves the wording with the direction.
- Without JavaScript, `<details>` sections that open on desktop only (A and C filters, B Top Picks) need a server-rendered `open` default; the mockups set it with a small script.
