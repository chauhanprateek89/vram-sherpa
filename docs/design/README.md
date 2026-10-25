# UI/UX overhaul: design (Gate 2, round 1 redo of 2)

The user rejected the first three directions (Basecamp, Altimeter, Expedition) and asked for "highly modern" ones: bento grids, glassmorphism or spatial UI, with no skeuomorphic elements, no physical-instrument metaphors and no standard flat design. These are the three replacements. The user picks one; round 2 then mocks up every screen and state of R-067 in it.

Gallery (private Artifact, before screenshots, the three directions side by side at both widths and themes, and their spec sheets): https://claude.ai/artifact/GpFFFNrx4duQTg63MYdQQQ
Local copy: `docs/design/gallery.html` (open it from this folder so the relative images resolve).
The rejected round is archived in `docs/design/rejected-r1/` (boards, mockups, fonts, old gallery and README).

## Index

| Direction | Live mockup (`?theme=dark` for dark; resize for 390 / 1280) | Spec sheet (type, layers, shadows, motion, contrast, logo, favicon) |
|---|---|---|
| D. Bento: a grid of tiles, answer tile biggest | `mockups/direction-d-home.html` | `mockups/direction-d-spec.html` |
| E. Glass: layered frosted panes over a colour field | `mockups/direction-e-home.html` | `mockups/direction-e-spec.html` |
| F. Spatial: windows at different heights above a floor | `mockups/direction-f-home.html` | `mockups/direction-f-spec.html` |

Renders (full page), per direction `x` in d, e, f: `direction-x-home-390-light.png`, `direction-x-home-390-dark.png`, `direction-x-home-1280-light.png`, `direction-x-home-1280-dark.png`.
Before (today's app): `before/results-{390,1280}-{light,dark}.png`.
Logo marks and favicons: `marks/direction-x-mark.svg`, `marks/direction-x-favicon.svg` (colours switch with `prefers-color-scheme`); drawn large and at 64, 32 and 16 px in each spec sheet.
Fonts (SIL OFL 1.1, one woff2 each, with licence): `fonts/` (Bricolage Grotesque for D, Outfit for E, Instrument Sans for F).

## What is live in the mockups

Hover and press on cards, Top Picks and buttons; the quantisation switcher with a sliding thumb; the gauge fill (press a quantisation button or Check compatibility; it uses htmx's own `.htmx-added` / `.htmx-request` classes); Filters (the count updates as boxes are ticked); How this is estimated (skeleton, then the five rows); keyboard focus rings; the theme toggle; example chips and Clear. D has no blur. E and F add pointer sheen or tilt on a fine pointer. Everything is off under `prefers-reduced-motion`.

## Measured on the boards (24 GB, 4k)

Where the first card starts, and in brackets where its verdict and margin end. Targets: 800 px at 1280, 1688 px at 390 (R-069).

| | 1280 px | 390 px | Top Picks (Q-1) | Blurred layers | Page height 1280 / 390 |
|---|---|---|---|---|---|
| Today | 1147 | 2800 (page is 540 px wide in a 390 px window) | above the cards | 0 | |
| D Bento | 624 (764) | 1058 (1278) | tiles under the answer; one collapsed row under the answer on a phone | 0 | 1246 / 2291 |
| E Glass | 481 (685) | 1019 (1244) | side rail in the list pane; collapsed row after the first card on a phone | 5 desktop, 3 phone | 1666 / 2315 |
| F Spatial | 668 (735) | 1048 (1275) | raised tier overlapping the answer window; collapsed row after the first card on a phone | 1 | 1539 / 2406 |

All three: `scrollWidth` equals 390 at 390 px; every button, chip, link, summary, checkbox label and quantisation button measures at least 44 px at 390 px (model-name links are padded to 44 px); no text under 14 px; fields are 16 px.

## Contrast

Token pairs are computed in each spec sheet (text 4.5:1, non-text 3:1, both themes, 56 to 64 checks per direction, all pass). On glass the pairs are measured against the worst backdrop colour, with the CSS `saturate()` applied, and against a nested inset sheet.

Rendered checks (text hidden, every pixel in each text box compared with the text colour): D lowest 6.19 (light) and 6.47 (dark); E with the backdrop forced to its worst colour, lowest 7.00 (light) and 5.52 (dark); F lowest 5.24 (light) and 7.10 (dark).

## Requirement coverage

R-068 identity, logo mark and favicon (Q-7), R-069 answer first, Top Picks placement and collapsed phone filters (Q-1, Q-3, Q-4), R-070 phone layout, R-071 form, R-072 card, switcher and gauge, R-073 breakdown (open, loading and loaded states are live), R-076 header and footer, R-077 contrast, focus and reduced motion, R-078 same figures, routes and controls. Round 2 covers the remaining states of R-067 (errors, empty and unseeded, model detail, how it works, GPU feedback, active filter chips).

## Where a requirement pushed back

- Contrast on glass (R-077): the dark glass pane at 70% opacity measured 3.64:1 for muted text on the brightest backdrop colour. It is 82% now (5.52:1). Muted text on the dark pane was also lightened.
- Blur on phones (E): 3 blurred layers on a phone instead of 5, by dropping blur from the answer pane and the footer under 900 px. F blurs only the header.
- Density (R-069): D reaches 764 px by putting the quantisation switcher on the title row inside each half-width card; without that the verdict would end near 830 px. F reaches 735 px by laying the card out in two columns on wide screens.
- Tap size (R-070): the four quantisation buttons share one row at 390 px (about 80 px each). Header nav needs two rows on a phone in all three, because brand, two links and the theme toggle do not fit in 358 px at 44 px targets.
- Motion (R-077): the gauge fill, thumb slide and pointer effects are all nice-to-have; under reduced motion each has a stated static result.

## Open points for the build

- Wording to approve with the direction (Q-2): Top Picks descriptions ("The strongest model that still leaves room to spare.", "Nearly as capable, and comfortable to run.", "The lightest model that still runs on your GPU."), the GPU placeholder "Type a GPU name" (the old one was cut off in a narrow field), the breakdown row labels (Model weights, Context memory (KV cache), Runtime overhead, Safety reserve, Total needed), and the help text "Pick your GPU or enter VRAM, then check." Errors, verdict labels and figures are unchanged.
- Without JavaScript, the `<details>` for Top Picks that open on wide screens only need a server-rendered `open`; the mockups set it with a small script. The quantisation thumb is progressive: without JS the pressed button carries its own fill.
- D and F use container queries and `:has()`. F's verdict-coloured cast shadow and D/E's checked-radio styling use `:has()`; the build should add a server-side verdict class to the card.
- E and F need solid fallbacks for `backdrop-filter` and `prefers-reduced-transparency`; they are in the mockups and the spec sheets give the colours.
- Each direction ships one font file of 32 to 77 KB (ADR-0002 allows four files and 200 KB). ADR-0002 still needs `static/fonts/*` added to package-data and versioned file names.
- The new tokens live inline in each mockup. For the chosen direction, round 2 moves them to `docs/design/tokens.css` and links it from every mockup.
