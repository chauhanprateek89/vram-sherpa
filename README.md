# VRAM Sherpa

MVP web app to estimate local LLM variant VRAM feasibility using a transparent estimation policy.

## Run locally (Docker)

```bash
docker compose up --build
```

App: `http://localhost:8000`

## Seed command

```bash
docker compose exec web python -m vramsherpa.seed
```

Or outside Docker (with env vars set):

```bash
python -m vramsherpa.seed
```

## Run tests

```bash
pip install -e .[dev]
pytest
```
