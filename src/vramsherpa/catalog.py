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


def _build_id_map(items: list[dict]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    used_ids: set[int] = set()
    next_generated_id = 1

    for item in items:
        raw_id = item.get("id")
        key = str(raw_id)
        if key in mapping:
            continue

        try:
            resolved_id = int(raw_id)
        except (TypeError, ValueError):
            while next_generated_id in used_ids:
                next_generated_id += 1
            resolved_id = next_generated_id
            next_generated_id += 1

        if resolved_id in used_ids:
            msg = f"Duplicate resolved id={resolved_id} for seed item id={raw_id}"
            raise SeedError(msg)

        mapping[key] = resolved_id
        used_ids.add(resolved_id)

    return mapping


def _resolve_foreign_id(raw_id: object, id_map: dict[str, int], label: str) -> int:
    key = str(raw_id)
    if key in id_map:
        return id_map[key]

    try:
        fallback_id = int(raw_id)
    except (TypeError, ValueError):
        msg = f"{label} references missing id={raw_id}"
        raise SeedError(msg) from None

    if fallback_id not in id_map.values():
        msg = f"{label} references missing id={raw_id}"
        raise SeedError(msg)
    return fallback_id


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


def _upsert_gpu(session: Session, item: dict, resolved_id: int) -> None:
    gpu = session.get(GPU, resolved_id)
    if gpu is None:
        gpu = GPU(id=resolved_id)
    gpu.vendor = str(item["vendor"])
    gpu.name = str(item["name"])
    gpu.vram_gb = float(item["vram_gb"])
    gpu.notes = _as_optional_text(item.get("notes"))
    session.add(gpu)


def _upsert_model(session: Session, item: dict, resolved_id: int) -> None:
    model = session.get(Model, resolved_id)
    if model is None:
        model = Model(id=resolved_id)
    model.name = str(item["name"])
    model.family = str(item["family"]).strip().title()
    model.params_b = float(item["params_b"])
    model.model_type = item.get("model_type")
    model.license = item.get("license")
    model.kv_gb_per_1k_ctx = float(item["kv_gb_per_1k_ctx"])
    model.sources = _as_optional_text(item.get("sources"))
    session.add(model)


def _upsert_variant(session: Session, item: dict, resolved_id: int, model_id: int) -> None:
    variant = session.get(Variant, resolved_id)
    if variant is None:
        variant = Variant(id=resolved_id)
    variant.model_id = model_id
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

    gpu_id_map = _build_id_map(gpu_items)
    model_id_map = _build_id_map(model_items)
    variant_id_map = _build_id_map(variant_items)

    for item in gpu_items:
        resolved_id = gpu_id_map[str(item.get("id"))]
        _upsert_gpu(session, item, resolved_id)
    session.flush()

    for item in model_items:
        resolved_id = model_id_map[str(item.get("id"))]
        _upsert_model(session, item, resolved_id)
    session.flush()

    model_ids = {model_id for (model_id,) in session.execute(select(Model.id)).all()}
    for item in variant_items:
        resolved_model_id = _resolve_foreign_id(item.get("model_id"), model_id_map, "Variant")
        if resolved_model_id not in model_ids:
            msg = f"Variant references missing model_id={item.get('model_id')}"
            raise SeedError(msg)
        resolved_variant_id = variant_id_map[str(item.get("id"))]
        _upsert_variant(session, item, resolved_variant_id, resolved_model_id)

    meta = session.get(CatalogMeta, "catalog_version")
    if meta is None:
        meta = CatalogMeta(key="catalog_version", value=catalog_version)
    else:
        meta.value = catalog_version
    session.add(meta)

    session.commit()
    return catalog_version
