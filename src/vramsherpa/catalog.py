from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from vramsherpa.models import GPU, CatalogMeta, Model, Variant

DEFAULT_DATA_DIR = Path(os.getenv("SEED_DATA_DIR", "data"))


class SeedError(RuntimeError):
    pass


def _as_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _read_seed(path: Path) -> tuple[str, list[dict]]:
    if not path.exists():
        msg = f"Seed file not found: {path}"
        raise SeedError(msg)

    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("catalog_version")
    items = payload.get("items")
    if not isinstance(version, str) or not isinstance(items, list):
        msg = f"Invalid seed file format: {path}"
        raise SeedError(msg)
    return version, items


def _upsert_gpu(session: Session, item: dict) -> None:
    gpu_id = str(item["id"])
    gpu = session.get(GPU, gpu_id)
    if gpu is None:
        gpu = GPU(id=gpu_id)
    gpu.vendor = str(item["vendor"])
    gpu.name = str(item["name"])
    gpu.vram_gb = float(item["vram_gb"])
    gpu.notes = _as_optional_text(item.get("notes"))
    session.add(gpu)


def _upsert_model(session: Session, item: dict) -> None:
    model_id = str(item["id"])
    model = session.get(Model, model_id)
    if model is None:
        model = Model(id=model_id)
    model.name = str(item["name"])
    model.family = str(item["family"])
    model.params_b = float(item["params_b"])
    model.model_type = _as_optional_text(item.get("model_type"))
    model.license = _as_optional_text(item.get("license"))
    model.kv_gb_per_1k_ctx = float(item["kv_gb_per_1k_ctx"])
    model.sources = _as_optional_text(item.get("sources"))
    session.add(model)


def _upsert_variant(session: Session, item: dict) -> None:
    variant_id = str(item["id"])
    variant = session.get(Variant, variant_id)
    if variant is None:
        variant = Variant(id=variant_id)
    variant.model_id = str(item["model_id"])
    variant.quant_bucket = str(item["quant_bucket"])
    variant.quant_label = str(item.get("quant_label") or item["quant_bucket"])
    variant.bits_effective = float(item["bits_effective"])
    variant.notes = _as_optional_text(item.get("notes"))
    variant.sources = _as_optional_text(item.get("sources"))
    variant.recommended = bool(item.get("recommended", False))
    session.add(variant)


def seed_catalog(session: Session, data_dir: Path = DEFAULT_DATA_DIR) -> str:
    gpu_version, gpu_items = _read_seed(data_dir / "seed_gpus.json")
    model_version, model_items = _read_seed(data_dir / "seed_models.json")
    variant_version, variant_items = _read_seed(data_dir / "seed_variants.json")

    versions = {gpu_version, model_version, variant_version}
    if len(versions) != 1:
        msg = f"Catalog versions do not match across seed files: {versions}"
        raise SeedError(msg)
    catalog_version = versions.pop()

    for item in gpu_items:
        _upsert_gpu(session, item)
    session.flush()

    for item in model_items:
        _upsert_model(session, item)
    session.flush()

    model_ids = {model_id for (model_id,) in session.execute(select(Model.id)).all()}
    for item in variant_items:
        model_id = str(item.get("model_id"))
        if model_id not in model_ids:
            msg = f"Variant references missing model_id={item.get('model_id')}"
            raise SeedError(msg)
        _upsert_variant(session, item)

    meta = session.get(CatalogMeta, "catalog_version")
    if meta is None:
        meta = CatalogMeta(key="catalog_version", value=catalog_version)
    else:
        meta.value = catalog_version
    session.add(meta)

    session.commit()
    return catalog_version
