# VRAM Sherpa: high-level design

2026-10-25 · Status: as built (main at 96572b1, last release tag v0.1.48)

## Purpose

VRAM Sherpa is a public, read-only web app. You pick a GPU or type in a VRAM amount and a context length, and it estimates which local LLM model variants (quantisations) will fit, using a published formula (Estimation Policy v0.1) and showing the numbers behind each verdict. It is for people choosing which open-weight model to run on their own hardware. It has no accounts and stores nothing that users enter.

## Components

| Component | Responsibility | Code | Talks to |
| --- | --- | --- | --- |
| App factory | Builds the FastAPI app, configures the DB engine, mounts `/static`, registers routes and the host guard. Fails at startup without `DATABASE_URL` | `src/vramsherpa/main.py` (`create_app`, `lifespan`) | Database layer, Settings |
| Settings | Reads `DATABASE_URL`, `APP_ENV` (default `dev`) and `ALLOWED_HOSTS` (comma list; blank means `*`) from the environment | `src/vramsherpa/config.py` | none |
| Host guard | HTTP middleware that returns 400 JSON for any `Host` not in `ALLOWED_HOSTS`, unless the list is `*`. It does not depend on `APP_ENV` | `src/vramsherpa/main.py` (`host_guard`) | Settings |
| Page routes | `GET /` (home with the default results: first GPU or 8 GB, 2048 tokens; ignores query params), `GET /results` (full page, or the `results_content.html` partial when `HX-Request: true`), `GET /models/{model_id}`, `GET /how-it-works` | `src/vramsherpa/main.py` (`home`, `results`, `model_detail`, `how_it_works`) | Estimation, ORM models, templates |
| HTMX partial routes | `GET /results/partials/{content,summary,list}`, and `GET /results/why/{variant_id}` for the per-variant breakdown | `src/vramsherpa/main.py` (`results_partial_*`, `variant_breakdown`) | Estimation, ORM models, templates |
| Health endpoint | `GET /healthz`: 200 `{"status":"ok","ready":true,"catalog_version",counts}` when the DB answers and the catalog is seeded, otherwise 503 | `src/vramsherpa/main.py` (`healthz`) | Database |
| Estimation engine | Pure functions: weights + KV cache + runtime overhead, a reserve of `max(0.75, 10%)`, and the Fits / Tight / Won't fit verdict | `src/vramsherpa/estimation.py` | none |
| Results shaping | Filters (family, quant bucket, min and max params, recommended only), sorting, one card per model (a model is hidden when no variant fits or is tight), top picks, filter chips, empty states | `src/vramsherpa/main.py` (`_build_results`, `_group_results_by_model`, `_results_context`) | Estimation |
| ORM models | Tables `gpus`, `models`, `variants`, `catalog_meta` | `src/vramsherpa/models.py` | Database layer |
| Database layer | SQLAlchemy 2 engine and session factory, plus `create_all` (creates missing tables, never alters existing ones) | `src/vramsherpa/database.py` | Postgres 16 (runtime), SQLite (tests) |
| Catalog seeder | Reads the three `data/seed_*.json` files, checks that their `catalog_version` values match, upserts by id, and writes `catalog_meta.catalog_version`. Runs as `python -m vramsherpa.seed` before uvicorn starts | `src/vramsherpa/catalog.py`, `src/vramsherpa/seed.py` | Database, `data/` (`SEED_DATA_DIR`, default `data`) |
| UI templates | Jinja2 server-rendered pages and partials. The footer shows the catalog version, the policy version and the last-updated date | `src/vramsherpa/templates/*.html` | Static assets |
| Static assets | `styles.css` and `app.js` (vanilla JS, 181 lines), cache-busted with `?v=ASSET_VERSION`. HTMX 1.9.12 and Google Fonts load from public CDNs in the browser | `src/vramsherpa/static/`, `src/vramsherpa/templates/base.html` | unpkg.com, fonts.googleapis.com (browser only) |
| Catalog ingest (offline) | Builds `data/seed_*.json` from the curated YAML, carries over models found by earlier discovery runs, discovers new HF models (`discovery.enabled: true`), can enrich them with HF metadata, generates Q4/Q5/Q8/FP16 variants, then validates the result | `tools/ingest_catalog.py`, `tools/catalog_sources.yaml` | huggingface.co API (build time only), `data/` |
| Catalog validator | Checks the seed files: matching versions, unique ids, required fields, bits per quant bucket, and that every variant has a model | `tools/validate_catalog.py` | `data/`, `tools/catalog_sources.yaml` |
| CI | Runs ruff check, ruff format --check, the catalog validator and pytest on PRs. `release.yml` reuses it through `workflow_call` | `.github/workflows/ci.yml` | GitHub Actions |
| Release | On push to main: CI, then build and push to GHCR (`vX.Y.N`, `sha-…`, `latest`), smoke-test the published image, then git tag and GitHub Release | `.github/workflows/release.yml` | GHCR, GitHub Releases |
| Catalog refresh | Runs weekly (Sun 08:00 UTC) and on manual dispatch: ingest, validate, then a PR on `automation/catalog-refresh` if the seed files changed | `.github/workflows/catalog-refresh.yml` | huggingface.co, GitHub PRs |
| Local stack | `db` (postgres:16-alpine, volume `pgdata`) and `web` (built from the Dockerfile, seeds then serves on `8000:8000`) | `docker-compose.yml`, `Dockerfile` | none |
| Production stack | `db`, `web` (GHCR image `${IMAGE_TAG:-latest}`, bound to `127.0.0.1:8000`, `--proxy-headers`, watchtower label) and `watchtower` (`nickfedor/watchtower:1.23.2`, `--label-enable`, polls every 300 s) | `docker-compose.prod.yml` | GHCR, Docker socket, cloudflared on the host |

## Data

- **Store:** PostgreSQL 16 in a container with the named volume `pgdata` (in both the local and the prod compose files). Tests use a temporary SQLite file per test (`tests/conftest.py`). CI starts a Postgres service and sets `TEST_DATABASE_URL`, but no code reads that variable.
- **Tables** (`src/vramsherpa/models.py`): `gpus` (id text PK, vendor, name, vram_gb, notes), `models` (id, name, family, params_b, model_type, license, gated, kv_gb_per_1k_ctx, sources as a JSON string), `variants` (id, model_id FK with cascade, quant_bucket, quant_label, bits_effective, notes, sources, recommended), `catalog_meta` (key/value, currently only `catalog_version`).
- **Seed data:** `data/seed_gpus.json`, `data/seed_models.json` and `data/seed_variants.json`, each shaped `{"catalog_version": "YYYY-MM-DD", "items": [...]}`. Current catalog: 2026-10-23, with 35 GPUs, 95 models and 380 variants. The files are generated by `tools/ingest_catalog.py` and committed. They are also copied into the image (`Dockerfile`), so every release carries its own catalog.
- **Persisted versus rebuilt:** the database holds no user data. It is rebuilt from the seed files every time the container starts. Seeding only upserts: rows that were dropped from the seed files stay in an existing volume until it is recreated.
- **Schema changes:** there are no migrations. `create_all` will not add columns to existing tables, so a new column breaks seeding on an old volume (`UndefinedColumn`). The fix is `docker compose down -v` or a manual `ALTER TABLE` (README, "Upgrading an existing database"). This happened when onboarding re-ran the local stack: the local volume had no `models.gated`.
- **Discovered models** live only in `data/seed_models.json` (flag `discovered: true`). `_load_discovered_models` reads them back on every ingest, so a refresh adds to the catalog and never drops earlier finds.

## Key flows

1. **Estimate request.** The browser sends `GET /results?vram_gb=…|gpu_id=…|gpu_search=…&context_tokens=…&family=…&quant_bucket=…`. `host_guard` checks the Host header. `results` calls `_results_page_context`, and `_resolve_available_vram` turns the GPU id, the search text (including VRAM hints such as "24GB") or the manual number into an available VRAM figure. `_build_results` queries variants joined to models with the filters applied, and `estimate_variant` scores each row. `_group_results_by_model` builds the cards, and the templates render `index.html`, or only `results_content.html` when the request comes from HTMX.
2. **HTMX interaction.** The form in `index.html` sends `hx-get` requests to `/results/partials/content`, which swaps `#results-shell`. The "why" buttons in `results_list.html` fetch `/results/why/{variant_id}` and show the formula breakdown from `estimate_breakdown`.
3. **Container start.** The compose command runs `python -m vramsherpa.seed` (which creates tables and seeds, printing `Seeded catalog_version=…`) and then uvicorn with `vramsherpa.main:create_app --factory`. If seeding fails, the container exits. `/healthz` returns 503 until the catalog rows exist.
4. **Catalog refresh.** The weekly workflow runs `ingest_catalog.py`: it loads the curated YAML, carries over earlier discoveries, discovers new models from the tracked HF orgs (min 50k downloads, at most 20 per org, exclusion patterns, `params_b_overrides`), generates variants and writes the seed files. It then runs `validate_catalog.py` and opens a "Catalog refresh YYYY-MM-DD" PR. Merging that PR is an ordinary merge to main, so it releases and deploys.
5. **Release and deploy.** A merge to main triggers `release.yml`: CI, then the version (`v<pyproject major.minor>.<git rev-list --count HEAD>`), fail if the tag exists, build and push to GHCR, and smoke-test the published image against Postgres. The smoke test checks that /healthz answers, that the seed log line appears, that `/` renders, that asset URLs are https behind `X-Forwarded-Proto: https`, and that an unknown Host gets 400. Then it pushes the git tag and creates the GitHub Release. On the homelab, watchtower sees the new `:latest` within 5 minutes and recreates `web`.

## Runtime and deploy

- **Local:** `.sdlc/init.sh` runs `docker compose -f docker-compose.yml up -d --build` and polls `http://localhost:8000/healthz` for up to 120 s. Without Docker: `pip install -e ".[dev]"` (conda `environment.yml`, Python 3.12, supported range `>=3.12,<3.14`), set `DATABASE_URL`, run `python -m vramsherpa.seed`, then `uvicorn vramsherpa.main:create_app --factory`.
- **Checks:** `ruff check .`, `ruff format --check .`, `python tools/validate_catalog.py --data-dir data --config tools/catalog_sources.yaml` and `pytest` (86 tests; the `node --check app.js` test is skipped when node is not installed). pre-commit runs the standard hygiene hooks plus ruff v0.9.10. The tools need `pyyaml`, which arrives through `pre-commit` in the dev extras.
- **Production:** the homelab runs `docker-compose.prod.yml` with a `chmod 600` `.env` (`APP_ENV`, `ALLOWED_HOSTS`, `PUBLIC_HOSTNAME`, `POSTGRES_PASSWORD`, optional `IMAGE_TAG` for rollback). The existing cloudflared systemd tunnel reaches `127.0.0.1:8000`. CI has no inbound path to the server. The GHCR package must be public.
- **Settings that matter:** `ALLOWED_HOSTS` is the only control on accepted hosts. `APP_ENV` only changes the startup error message, and the code never reads `PUBLIC_HOSTNAME`. So the public DNS name must be in `ALLOWED_HOSTS`, or the host guard rejects tunnel traffic with 400. `docs/RELEASE_AND_DEPLOY.md` says otherwise. `--proxy-headers --forwarded-allow-ips=*` must stay on the prod command, or asset URLs fall back to http:// and the browser blocks them as mixed content.
- **Caching:** HTML responses send no `Cache-Control` header (checked on `GET /`). Static assets are cache-busted by the hand-bumped `ASSET_VERSION` (`20261024a`).
- **Version visible on the live site:** there is no release version in any response. The marker closest to it is `styles.css?v=<ASSET_VERSION>` in the HTML, which changes only when a developer bumps it. `/healthz` and the footer show `catalog_version`, which changes only when the catalog does. The release tag exists only as the image label `org.opencontainers.image.version`.
- **Live URL:** unknown. The public hostname has not been recorded in the repo.

## Decisions

| ADR | Decision |
| --- | --- |
| (none yet) | Decisions taken before onboarding are recorded in `docs/RELEASE_AND_DEPLOY.md` (pull-based CD, watchtower fork, homelab-only deploy) and in `docs/decisions.md` from now on. |
