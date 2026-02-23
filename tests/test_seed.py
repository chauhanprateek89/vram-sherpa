from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from vramsherpa.catalog import seed_catalog
from vramsherpa.database import get_engine
from vramsherpa.models import GPU, CatalogMeta, Model, Variant


def test_tables_create_successfully(db_session: Session) -> None:
    inspector = inspect(get_engine())
    table_names = set(inspector.get_table_names())
    assert {"gpus", "models", "variants", "catalog_meta"}.issubset(table_names)


def test_seed_loads_and_is_idempotent(db_session: Session) -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"

    version_1 = seed_catalog(db_session, data_dir=data_dir)
    counts_1 = (
        db_session.scalar(select(func.count(GPU.id))),
        db_session.scalar(select(func.count(Model.id))),
        db_session.scalar(select(func.count(Variant.id))),
    )

    version_2 = seed_catalog(db_session, data_dir=data_dir)
    counts_2 = (
        db_session.scalar(select(func.count(GPU.id))),
        db_session.scalar(select(func.count(Model.id))),
        db_session.scalar(select(func.count(Variant.id))),
    )
    family_count = db_session.scalar(select(func.count(func.distinct(Model.family))))

    assert version_2 == version_1
    assert version_1
    assert counts_1 == counts_2
    assert counts_1[0] >= 20
    assert counts_1[2] >= 50
    assert family_count is not None
    assert family_count >= 3

    meta = db_session.get(CatalogMeta, "catalog_version")
    assert meta is not None
    assert meta.value == version_1
