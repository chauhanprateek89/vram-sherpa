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

- `DATABASE_URL`: Runtime database URL. Defaults to `sqlite+pysqlite:///./vramsherpa.db`.
- `TEST_DATABASE_URL`: Optional test database URL. If unset, tests use a temporary SQLite database.
- `APP_ENV`: Optional app mode (`dev` by default).
- `ALLOWED_HOSTS`: Optional comma-separated host allowlist.

Examples:

```bash
export DATABASE_URL="postgresql+psycopg://vramsherpa:vramsherpa@localhost:5432/vramsherpa"
export TEST_DATABASE_URL="postgresql+psycopg://vramsherpa:vramsherpa@localhost:5432/vramsherpa_test"
```

## Run tests and checks

```bash
ruff check .
ruff format --check .
pytest
```

## Seed data

Inside Docker:

```bash
docker compose exec web python -m vramsherpa.seed
```

Outside Docker (with `DATABASE_URL` set as needed):

```bash
python -m vramsherpa.seed
```

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
