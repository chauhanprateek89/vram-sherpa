from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from vramsherpa.config import get_settings

Base = declarative_base()

_engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def configure_engine(database_url: str | None = None) -> Engine:
    global _engine, SessionLocal
    url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        return configure_engine()
    return _engine


def create_db_and_tables() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_session() -> Generator[Session, None, None]:
    local_session = SessionLocal
    if local_session is None:
        configure_engine()
        local_session = SessionLocal
    assert local_session is not None
    with local_session() as session:
        yield session
