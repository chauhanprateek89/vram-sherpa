# ADR-0017: Turn off htmx's history cache, so every Back and Forward reloads from the address

Date: 2026-11-02 · Status: proposed · Requirements: R-088, R-081, R-086, R-089, R-090 · Decisions: Q-33 (Gate 1, 2026-11-02) · Constitution: rule 11 · Amends: ADR-0008 (fix 2 and its alternatives table)

## Context

Before each push, htmx 1.9.12 saves a snapshot of `<body>` and the entry's URL to `localStorage` under `htmx-history-cache`. With verdicts in the address, that snapshot holds the `verdict`, `page` and `page_size`. R-088 says: "THE SYSTEM SHALL keep the verdict, page and page size in the address only, and SHALL NOT store them in a cookie, `localStorage` or on the server". The default cache breaks that requirement.

The cache also conflicts with this change in other ways:

- **R-086:** "WHEN the user comes back to a card after moving to another page, THE SYSTEM SHALL open it on its default variant". A snapshot keeps the pressed quantisation.
- **R-090:** a snapshot taken while `aria-busy="true"` or an error message is showing would restore them.
- **ADR-0008's known issue:** on a cache hit, listeners that `app.js` attached to individual elements are lost, so the theme toggle and the GPU box stop working on a restored page.

ADR-0008 rejected `historyCacheSize: 0` because it "turns every Back into a server round trip". R-088 did not exist then. Each Back now costs one full page load, which for the reference query is a smaller response than today's because it carries at most 10 cards.

## Decision

1. **`base.html`'s meta tag becomes `{"refreshOnHistoryMiss":true,"historyCacheSize":0}`.**
2. Checked against htmx 1.9.12's `dist/htmx.js`:
   - `saveToHistoryCache` runs before each push and at the start of `restoreHistory`, and with a size of 0 or less it removes `htmx-history-cache` and returns, so a returning visitor's old cache is purged at their first in-place refresh or Back.
   - `saveCurrentPageToHistory` still calls `history.replaceState({htmx: true}, …)`, so entries remain htmx entries and popstate reaches `restoreHistory`.
   - With nothing cached, `restoreHistory` takes the miss branch, and `refreshOnHistoryMiss` calls `window.location.reload(true)`.
3. **ADR-0008's fixes 1, 3 and 4 stay.** Fix 3 remains the safety net if the meta tag is ever lost, though in that case the cache would also come back on, and the guard test below catches it.
4. **Nothing else is added for restored state.** With no restored snapshots, there is no `htmx:beforeHistorySave` cleanup to write, and the existing `htmx:historyRestore` listeners in `app.js` become unused but harmless; the build may leave them.

Guard (pytest, rule 19): `tests/test_history.py` currently asserts the exact meta content. It changes to the new value.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Keep the cache and reword R-088 | Reopens a Gate 1 requirement, and leaves the R-086, R-090 and re-wiring problems to solve one by one. Offered at Gate 2 as the way to avoid it (see below). |
| `hx-history="false"` on an element | Skips saving but does not purge an existing cache, and `restoreHistory` would still read it. Needs an element on every page to do what one config value does. |
| `htmx:beforeHistorySave` to reset cards and strip busy state | Fixes R-086 and R-090 but still stores the view in `localStorage`, against R-088, and leaves the re-wiring issue. |
| `historyEnabled: false` | Stops pushes too. R-088 needs history entries. |

## Consequences

- R-081 and R-088's Back and Forward criteria hold, and the "with or without a history miss" cases become one case. `localStorage.removeItem('htmx-history-cache')` in R-081's test setup is now a harmless no-op.
- ADR-0008's out-of-scope issue, dead listeners on a cache hit, stops occurring.
- Back and Forward within `/results` are now a full page load, so a brief repaint replaces an instant restore. The browser restores scroll by its own rules; no requirement fixes a scroll position on Back. **The user sees this change, so it is raised at Gate 2.**
- Text typed in the form and not submitted does not survive Back or Forward. That matches the address, which R-088 makes the source of truth.
- `UI: no`. No mockup.
