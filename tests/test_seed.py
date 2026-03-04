from __future__ import annotations

import json
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


def test_seed_normalizes_list_and_string_text_fields(db_session: Session, tmp_path: Path) -> None:
    payloads = {
        "seed_gpus.json": {
            "catalog_version": "2026-03-03",
            "items": [
                {
                    "id": "gpu_test_1",
                    "vendor": "TestVendor",
                    "name": "TestGPU",
                    "vram_gb": 8,
                    "notes": ["alpha", "beta"],
                }
            ],
        },
        "seed_models.json": {
            "catalog_version": "2026-03-03",
            "items": [
                {
                    "id": "model_test_1",
                    "name": "TestModel",
                    "family": "test",
                    "params_b": 1.5,
                    "model_type": "chat",
                    "license": "apache-2.0",
                    "kv_gb_per_1k_ctx": 0.1,
                    "sources": ["source-a", "source-b"],
                }
            ],
        },
        "seed_variants.json": {
            "catalog_version": "2026-03-03",
            "items": [
                {
                    "id": "variant_test_1",
                    "model_id": "model_test_1",
                    "quant_bucket": "q4",
                    "quant_label": "Q4_K_M",
                    "bits_effective": 4.5,
                    "notes": "already text",
                    "sources": ["variant-source"],
                    "recommended": True,
                }
            ],
        },
    }

    for filename, payload in payloads.items():
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

    seed_catalog(db_session, data_dir=tmp_path)

    gpu = db_session.get(GPU, "gpu_test_1")
    model = db_session.get(Model, "model_test_1")
    variant = db_session.get(Variant, "variant_test_1")

    assert gpu is not None
    assert model is not None
    assert variant is not None
    assert gpu.notes == json.dumps(["alpha", "beta"])
    assert model.sources == json.dumps(["source-a", "source-b"])
    assert variant.notes == "already text"
    assert variant.sources == json.dumps(["variant-source"])
