# ADR-0018: Verdict, page and size controls are plain links that htmx refreshes in place, with feedback outside the swap and focus moved after it

Date: 2026-11-02 · Status: proposed · Requirements: R-085, R-087, R-090, R-091, R-032, R-078, R-080 · Decisions: Q-36, Q-37, Q-38 (Gate 1, 2026-11-02) · Constitution: rules 9, 10 and 11

## Context

R-090 requires each control to be a plain link without JavaScript, and an in-place refresh through `/results/partials/content` with it, leaving the form's fields alone. While a fetch runs, the current view stays with `aria-busy="true"` and `Loading models`. On failure, the view and the address stay and `Could not load the models. Try again.` appears in `role="alert"`. R-091 moves focus to the list's heading after a load, keeps it on the control after a collapse, and brings the heading into view without animation under reduced motion.

Findings from the code and from htmx 1.9.12's source shape the mechanism:

- `htmx:afterSettle` fires once for each new top-level element in the swap, not once per request. `#results-shell` swaps two children, so a focus handler would run twice.
- For a non-boosted `<a hx-get>`, htmx calls `preventDefault()` before evaluating any trigger filter, so `click[!ctrlKey]` would leave a modified click doing nothing at all.
- `hx-sync` is read with `getClosestAttributeValue`, so placing it on `#results-shell` would also catch the breakdown requests inside it.
- A `4xx` or `5xx` is not swapped, and htmx pushes history only in its swap path, so the address stays as R-090 requires, without any extra handling.
- By the time `htmx:afterRequest` fires after a swap, the link that made the request has left the document. htmx then fires it again on the closest ancestor still present, so a listener on `document.body` still hears it.
- `syncBoxVram` runs after every settle. If the address names a box number, it would overwrite text the user has typed and not submitted, which R-090 forbids.

## Decision

1. **Each control is an `<a>` with a real `href`.** One Jinja macro in `components.html` emits `href="/results?<q>"`, `hx-get="/results/partials/content?<q>"`, `hx-target="#results-shell"`, `hx-sync="#results-shell:replace"` and `data-view-link="verdict|collapse|size|page"` (`<q>` from ADR-0014). No `hx-push-url`: the partial's `HX-Push-Url` header pushes one entry whose path is `/results`.
2. **The form gets the same `hx-sync`.** The latest action wins, and an older response cannot land over a newer one or add a stray history entry. The attribute goes on the links and the form only.
3. **Inert states are not links.** A verdict at 0, `Previous` on page 1, `Next` on the last page and the current page number are `<span>`s, never focusable (R-085, R-087, R-091). The selected verdict stays a link, because selecting it again collapses. The current page size stays a link, as R-087 states.
4. **Feedback lives outside the swap.** `index.html` gains, as a sibling of `#results-shell`, one `#results-feedback` holding `<p id="results-loading" role="status" data-text="Loading models">` and `<p id="results-error" role="alert" data-text="Could not load the models. Try again.">`. Both are empty until a script fills them. This is what a live region needs to announce, and it survives every swap. They are empty without JavaScript, so nothing shows. The strings stay in the template, so tests can read them without a browser.
5. **Feedback is anchored to the viewport, not the page flow.** After `Next` under a long list, the top of the results may be off-screen, and a message there would not be seen. It also must not push the headline down, which would break R-069's limits. Its look is for the G12 mockup.
6. **Listeners, all on `document.body` and all filtered to `event.detail.target.id === "results-shell"`,** so breakdown requests are untouched:
   - `htmx:beforeRequest`: set `aria-busy="true"` on `#results-shell`, fill `#results-loading`, empty `#results-error`, and note whether `event.target` has `data-view-link`, and which kind.
   - `htmx:afterRequest`: remove `aria-busy`, empty `#results-loading`. With `hx-sync` replace, the aborted request's `afterRequest` fires inside `xhr.abort()`, before the new request's `beforeRequest`, so the order holds.
   - `htmx:responseError` and `htmx:sendError`: fill `#results-error`. `htmx:sendAbort` is deliberately not one of them, because an aborted request is not a failure.
   - `htmx:afterSettle`, once per request through a one-shot flag: run the focus step.
7. **Focus step.** After a `verdict`, `size` or `page` link: focus `#list-heading` (an `<h2 tabindex="-1">`) with `{preventScroll: true}`, then `scrollIntoView({block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth"})`. The heading's text, such as `Fits: showing 6-10 of 71 models`, is the announcement, so no extra live region is needed. After a `collapse` link: focus the control with the same `data-verdict-control` value, which is still a link because the count has not changed. Form-driven refreshes do not move focus, because the form is not swapped. The heading needs `scroll-margin-top`, because `.top` is sticky at wider widths.
8. **`syncVramInvalid` and `syncBoxVram` skip responses to a `data-view-link` request.** The address's hardware has not changed, and the form's fields must stay as typed (R-090). The typed text does not reach the server either, because the link's URL was built from the address (R-090).
9. **A modified click opens a new tab.** A capture-phase click listener calls `stopPropagation()` when Ctrl, Cmd or Shift is held on a `[data-view-link]`, so htmx never sees the click and the browser follows the `href`.
10. **Without JavaScript nothing is needed beyond the `href`.** Each link is a full page of the same view, through the same templates (ADR-0014 point 7).

## Alternatives considered

| Option | Why not |
| --- | --- |
| `hx-boost` on a container | Requests the `href`, so the fragment comes from `/results`, not `/results/partials/content` as R-090 states; it would also push whatever URL it requested and boost other links in the shell. |
| `hx-trigger="click[!ctrlKey&&!metaKey]"` | htmx 1.9.12 calls `preventDefault()` first, so a modified click would do nothing. |
| `<button hx-get>` for the controls | Not a link without JavaScript, against R-090 and R-091. |
| Feedback inside `#results-shell` | Replaced on each swap, so unreliable as a live region. After `Next` it would sit off-screen. |
| A separate `aria-live` announcement of the new heading | Two announcements of the same text once focus lands on the heading. |
| `hx-indicator` for the loading state | Sets a class, not `aria-busy`, and gives no live-region text. The listeners are still needed. |

## Consequences

- Every control works by keyboard as a normal link, in on-screen order (R-091), and can be opened in a new tab or copied.
- The `Cache-Control` and `Vary` middleware covers all new HTML with no change (rule 9, R-080).
- `app.js` and `styles.css` change, so `ASSET_VERSION` is bumped (rule 21).
- `UI: yes`. Controls, feedback and focus styles need approved mockups for G3 to G5, G12 and G13 (rule 10).
