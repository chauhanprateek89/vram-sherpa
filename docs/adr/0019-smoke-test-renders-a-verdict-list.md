# ADR-0019: The release smoke test adds a render check of a verdict list

Date: 2026-11-02 · Status: proposed · Requirements: R-062, R-084 · Decisions: Q-42 (Gate 1, 2026-11-02) · Constitution: rules 18 and 20 · Extends: ADR-0013

## Context

After R-084, `/results` with no verdict shows the gist, which holds no card. ADR-0013's check of `/results?vram_gb=24&context_tokens=4096` still passes, because `id="summary-banner"` and `Manual VRAM (24 GB)` stay in the gist, but it no longer exercises card rendering. At Gate 1 the user chose to add a check (Q-42), and R-062 states it. Rule 18 lets checks be added, never removed or weakened.

## Decision

1. **Every existing check stays unchanged,** including ADR-0013's.
2. **One check is added** to `.github/workflows/release.yml`, directly after ADR-0013's, against the same published digest: `GET http://127.0.0.1:8000/results?vram_gb=24&context_tokens=4096&verdict=fits`, read with `curl -sS -m 10 -o <file> -w '%{http_code}'` (no `-f`, no `-L`, as in ADR-0013). The release fails unless the status is exactly 200 and the body contains `data-model-card`. On failure it prints an `::error::` naming the status and exits 1, before any tag exists (R-082's order is unchanged).
3. **The web command is not touched** (rule 20).
4. **`tests/test_r082_release_order.py`** (`test_smoke_checks_unchanged`) gains `verdict=fits` and `data-model-card` as required needles, so the check cannot be dropped silently.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Point ADR-0013's check at the list URL | Replaces a check (rule 18), and the gist would lose its own render check. |
| Match a model name or a count | Ties the release to catalog contents, which change weekly. `data-model-card` is stable. |
| Check for `<verdict>: showing` | Rewordable text (Q-41). `data-model-card` is what `app.js` reads, so it cannot be reworded away. |

## Consequences

- A release that seeds, queries and estimates but cannot render a card now fails.
- If a future catalog had no model that fits 24 GB at 4k, the list would fall back to the gist and the release would fail. A catalog like that is broken, so failing is right.
- The task that changes `release.yml` is CI and deploy configuration, so it routes to Opus, and merging needs the user's sign-off (rule 6).
- `UI: no`.
