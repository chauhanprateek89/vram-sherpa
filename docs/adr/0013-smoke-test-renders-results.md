# ADR-0013: The release smoke test adds a render check of `/results`

Date: 2026-10-28 · Status: proposed · Requirements: R-062, R-083, R-013 · Decisions: Q-23 (Gate 1, 2026-10-28) · Constitution: rules 18 and 20

## Context

The release smoke test in `.github/workflows/release.yml` ("Smoke test the published image") requests `GET /?vram_gb=24&context_tokens=4096` and fails the release if it does not succeed. Until now `/` rendered results, so that check proved the published image could query the seeded catalog and run the estimation. After R-083, `/` ignores its parameters and renders the landing page, which computes nothing (ADR-0009). The check still passes, but it covers less. Rule 18 lets checks be added, never removed or weakened. At Gate 1 the user chose to add a `/results` render check in this change (Q-23), and R-062 now states it.

## Decision

1. **Keep every existing check unchanged,** including the `/?vram_gb=24&context_tokens=4096` request, the asset-URL scan of `/` behind `X-Forwarded-Proto: https`, the `/static/` fetch, the `Cache-Control` check on `/`, and the 400 for `Host: evil.example.com`. Its error text `Results page did not render.` may be reworded to `Home page did not render.`, since the check itself is the same.
2. **Add one check,** after the existing `/` request and before the asset checks: request `http://127.0.0.1:8000/results?vram_gb=24&context_tokens=4096` against the same published digest, and fail the release unless all of these hold:
   - the status is exactly 200, read with `curl -sS -m 10 -o <file> -w '%{http_code}'`, without `-L`. `curl -f` alone is not enough, because it treats a 3xx as success, and a regression that redirected this URL to `/` would pass;
   - the body contains `id="summary-banner"`;
   - the body contains `Manual VRAM (24 GB)`.
   On failure it prints `::error::Results page did not render.` and the status, and exits 1 before any tag is created (R-062, R-082's order is unchanged).
3. **The web command is not touched,** so the smoke container's command stays identical to `docker-compose.prod.yml` (rule 20).
4. **The release-order test** (`tests/test_r082_release_order.py`, `test_smoke_checks_unchanged`) gains `/results?vram_gb=24&context_tokens=4096`, `summary-banner` and `Manual VRAM (24 GB)` as required needles, so the new check cannot be dropped silently later.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Change the existing `/?…` request to `/results?…` | Replaces a check rather than adding one (rule 18), and drops the only request that proves `/` renders. |
| Check only the status of `/results?…` | A page that renders the default note or the empty state at 200 would pass. The `Manual VRAM (24 GB)` text proves the query reached the estimation with the right hardware. |
| Match a model name or count in the body | Ties the release to the catalog contents, which change weekly through `catalog-refresh.yml`. The hardware label and the summary id are stable. |
| Follow redirects (`-L`) | Would let ADR-0010's redirect turn a broken `/results` into a passing landing page. |

## Consequences

- The release again proves the image can seed, query and estimate, now through `/results`.
- `release.yml` is CI and deploy configuration: the task that changes it is routed to Opus and needs the user's sign-off at merge like any other change (rule 6).
- `UI: no`.
