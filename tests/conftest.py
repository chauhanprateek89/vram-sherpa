from __future__ import annotations

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
        database_url=test_database_url,
        app_env="test",
        allowed_hosts=("testserver", "localhost", "127.0.0.1"),
    )
    return create_app(settings)


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
