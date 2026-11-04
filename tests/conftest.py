from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from vramsherpa import database
from vramsherpa.catalog import seed_catalog
from vramsherpa.config import Settings
from vramsherpa.main import create_app


@pytest.fixture()
def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


@pytest.fixture()
def test_database_url(tmp_path: Path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'test.db'}"


@pytest.fixture()
def app(test_database_url: str):
    settings = Settings(
        database_url=None,
        app_env="test",
        allowed_hosts=("testserver", "localhost", "127.0.0.1"),
    )
    return create_app(settings, database_url=test_database_url)


@pytest.fixture()
def db_session(app) -> Session:
    engine = database.get_engine()
    database.Base.metadata.drop_all(bind=engine)
    database.Base.metadata.create_all(bind=engine)

    local_session = database.SessionLocal
    assert local_session is not None
    with local_session() as session:
        yield session


@pytest.fixture()
def seeded_session(db_session: Session, data_dir: Path) -> Session:
    seed_catalog(db_session, data_dir=data_dir)
    return db_session


_CARD_NAMES = re.compile(r"(?s)data-model-card.*?<h3>\s*<a [^>]*>([^<]+)")


@pytest.fixture()
def walk_verdict():
    """Read every page of one verdict at ``page_size=10``; return the page bodies (R-078).

    Stops at a page with fewer than 10 cards, or one whose cards equal the previous page's
    (a page past the end shows the last page, so an exact multiple of 10 would never stop).
    """

    def walk(client, query: str, verdict: str = "fits", path: str = "/results") -> list[str]:
        query = query.removeprefix("?").removeprefix(f"{path}?")
        bodies: list[str] = []
        previous: list[str] | None = None
        page = 1
        while True:
            url = f"{path}?{query}&verdict={verdict}&page_size=10&page={page}"
            response = client.get(url)
            assert response.status_code == 200, url
            names = _CARD_NAMES.findall(response.text)
            if names == previous:
                break
            bodies.append(response.text)
            if response.text.count("data-model-card") < 10:
                break
            previous = names
            page += 1
        return bodies

    return walk
