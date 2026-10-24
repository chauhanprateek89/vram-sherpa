# VRAM Sherpa

A public, read-only web app that estimates which local LLM variants fit a GPU's VRAM, showing the numbers behind each verdict.

Read before changing anything:

- `docs/constitution.md`: rules no change may break
- `docs/requirements.md`: what the app must do, by R-number
- `docs/HLD.md`: how it is built
- `docs/decisions.md`: decisions taken so far
- `.sdlc/STATUS.md`: where current work stands

Changes go through the sdlc pipeline (`/sdlc:orchestrate`): features on the Full track, fixes on the Quick track. `/sdlc:status` shows where things stand.

## Commands

| Task | Command |
| --- | --- |
| Run locally | `.sdlc/init.sh` (stop: `docker compose -f docker-compose.yml down`) |
| Test | `pytest` (after `pip install -e ".[dev]"`) |
| Lint | `ruff check . && ruff format --check . && python tools/validate_catalog.py --data-dir data --config tools/catalog_sources.yaml` |
| Format | `ruff format . && ruff check --fix .` |
| Rebuild catalog | `python tools/ingest_catalog.py --config tools/catalog_sources.yaml --output-dir data` |

## Conventions

- Python 3.12 (`>=3.12,<3.14`), FastAPI, Jinja2 server-rendered, HTMX 1.9 plus vanilla `app.js` (no SPA framework), SQLAlchemy 2, Postgres 16. Ruff line length 100.
- Code lives in `src/vramsherpa/`. Tests in `tests/test_*.py` use a temporary SQLite DB and the real `data/` seeds.
- The estimation formula (`estimation.py`) and `/how-it-works` must agree. A formula change bumps `ESTIMATION_POLICY_VERSION`.
- Never edit `data/seed_*.json` by hand. Change `tools/catalog_sources.yaml` and re-run the ingest. All three files share one `catalog_version`.
- No migrations: `create_all` never alters tables. A new column needs `docker compose down -v` locally and a note for the server.
- When you change `styles.css` or `app.js`, bump `ASSET_VERSION` in `main.py`.
- Merging to main releases and deploys to the homelab. The host guard reads only `ALLOWED_HOSTS`; `APP_ENV` and `PUBLIC_HOSTNAME` do not affect it.
- `docs/archive/agent.md` (gitignored, local only) is the original MVP brief. Where it and the code disagree, the code and these docs win.
