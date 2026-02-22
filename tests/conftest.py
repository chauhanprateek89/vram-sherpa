from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import database
from vramsherpa.catalog import seed_catalog


@pytest.fixture(scope="session")
def test_database_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        return url
    db_dir = tmp_path_factory.mktemp("db")
    return f"sqlite+pysqlite:///{db_dir / 'test.db'}"


@pytest.fixture()
def db_session(test_database_url: str) -> Session:
    database.configure_engine(test_database_url)
    engine = database.get_engine()
    database.Base.metadata.drop_all(bind=engine)
    database.Base.metadata.create_all(bind=engine)

    local_session = database.SessionLocal
    assert local_session is not None
    with local_session() as session:
        yield session


@pytest.fixture()
def seeded_session(db_session: Session) -> Session:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    seed_catalog(db_session, data_dir=data_dir)
    return db_session


@pytest.fixture()
def client(seeded_session: Session) -> TestClient:
    from vramsherpa.main import app

    with TestClient(app) as test_client:
        yield test_client
