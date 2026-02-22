from __future__ import annotations

from sqlalchemy import func, select

from vramsherpa import database
from vramsherpa.catalog import seed_catalog
from vramsherpa.models import GPU, Model, Variant


def main() -> None:
    database.configure_engine()
    database.create_db_and_tables()
    assert database.SessionLocal is not None
    with database.SessionLocal() as session:
        version = seed_catalog(session)
        gpu_count = session.scalar(select(func.count(GPU.id))) or 0
        model_count = session.scalar(select(func.count(Model.id))) or 0
        variant_count = session.scalar(select(func.count(Variant.id))) or 0
    print(
        f"Seeded catalog_version={version} with "
        f"{gpu_count} GPUs, {model_count} models, {variant_count} variants"
    )


if __name__ == "__main__":
    main()
