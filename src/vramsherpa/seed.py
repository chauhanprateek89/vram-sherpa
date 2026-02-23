from __future__ import annotations

from sqlalchemy import func, select

from vramsherpa import database
from vramsherpa.catalog import seed_catalog
from vramsherpa.config import get_settings
from vramsherpa.models import GPU, Model, Variant


def main() -> None:
    settings = get_settings()
    if not settings.database_url:
        msg = "DATABASE_URL must be set to run the seed command."
        raise RuntimeError(msg)

    database.configure_engine(settings.database_url)
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
