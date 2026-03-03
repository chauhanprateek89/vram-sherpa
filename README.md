# VRAM Sherpa

MVP web app to estimate local LLM variant VRAM feasibility using a transparent estimation policy.

## Python version policy

This project supports Python `>=3.12,<3.14`.

Python `3.14` is intentionally excluded for now because parts of the ecosystem (especially binary wheels
for dependencies used in web/database stacks) can lag behind a new Python release. Pinning the supported
range keeps local dev, CI, and Docker reproducible.

## Local setup (conda + editable install)

```bash
conda env create -f environment.yml
conda activate vram-sherpa
pip install -e ".[dev]"
```

## Environment variables

- `DATABASE_URL`: Runtime database URL. Required when `APP_ENV` is not `test`.
- `APP_ENV`: Optional app mode (`dev` by default).
- `ALLOWED_HOSTS`: Optional comma-separated host allowlist.

Examples:

```bash
export DATABASE_URL="postgresql+psycopg://vramsherpa:vramsherpa@localhost:5432/vramsherpa"
```

## Run tests and checks

```bash
ruff check .
ruff format --check .
pytest
```

Tests use SQLite by default (temporary file DB, no env vars required). Docker runs the app with Postgres.

## Seed data

Inside Docker:

```bash
docker compose exec web python -m vramsherpa.seed
```

Outside Docker (with `DATABASE_URL` set as needed):

```bash
python -m vramsherpa.seed
```

## Catalog refresh tooling (offline/build-time)

Curated catalog inputs live in:

- `tools/catalog_sources.yaml`

Generate seed files from curated sources:

```bash
python tools/ingest_catalog.py --config tools/catalog_sources.yaml --output-dir data
```

Optional Hugging Face metadata enrichment (for configured model repos):

```bash
python tools/ingest_catalog.py --config tools/catalog_sources.yaml --output-dir data --with-hf-metadata
```

Validate generated catalog files:

```bash
python tools/validate_catalog.py --data-dir data --config tools/catalog_sources.yaml
```

The ingest script always writes:

- `data/seed_gpus.json`
- `data/seed_models.json`
- `data/seed_variants.json`

Each file follows:

```json
{ "catalog_version": "YYYY-MM-DD", "items": [ ... ] }
```

## Scheduled catalog PR workflow

GitHub Actions workflow: `.github/workflows/catalog-refresh.yml`

- Triggers weekly (Sunday UTC) and manual dispatch.
- Runs ingest + validation before opening a PR.
- Opens a PR only when `data/seed_*.json` files changed.
- PR title format: `Catalog refresh YYYY-MM-DD`.
- PR body includes refreshed counts for GPUs/models/variants.

## Run with Docker Compose

```bash
docker compose up --build
```

Run tests in the container (installs dev extras first):

```bash
docker compose exec web sh -c 'pip install .[dev] && pytest'
```

Services:

- `web`: FastAPI app on `http://localhost:8000`
- `db`: Postgres 16

The web container seeds catalog data on startup and exposes a health endpoint at `/healthz`.
