from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_QUANT_BITS = {
    "Q4": 4.5,
    "Q5": 5.5,
    "Q8": 8.5,
    "FP16": 16.0,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON object must be a dict in {path}")
    return payload


def _validate_envelope(
    name: str, payload: dict[str, Any], errors: list[str]
) -> list[dict[str, Any]]:
    version = payload.get("catalog_version")
    if not isinstance(version, str) or not version:
        errors.append(f"{name}: catalog_version must be a non-empty string")
    else:
        try:
            date.fromisoformat(version)
        except ValueError:
            errors.append(f"{name}: catalog_version must be YYYY-MM-DD (got {version!r})")

    items = payload.get("items")
    if not isinstance(items, list):
        errors.append(f"{name}: items must be a list")
        return []

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"{name}[{idx}]: item must be an object")
            continue
        out.append(item)
    return out


def _expect_non_empty_string(
    item: dict[str, Any],
    field: str,
    label: str,
    errors: list[str],
) -> str | None:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}: {field} must be a non-empty string")
        return None
    return value


def _expect_number(item: dict[str, Any], field: str, label: str, errors: list[str]) -> float | None:
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: {field} must be a number")
        return None
    return float(value)


def _expect_sources_list(item: dict[str, Any], label: str, errors: list[str]) -> None:
    sources = item.get("sources")
    if not isinstance(sources, list):
        errors.append(f"{label}: sources must be a list")
        return

    for src_idx, source in enumerate(sources):
        if not isinstance(source, str) or not source.strip():
            errors.append(f"{label}: sources[{src_idx}] must be a non-empty string")


def _load_expected_quant_bits(config_path: Path | None) -> dict[str, float]:
    if config_path is None or not config_path.exists():
        return dict(DEFAULT_QUANT_BITS)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return dict(DEFAULT_QUANT_BITS)

    catalog = payload.get("catalog")
    if not isinstance(catalog, dict):
        return dict(DEFAULT_QUANT_BITS)

    raw_map = catalog.get("quant_bucket_bits_effective")
    if not isinstance(raw_map, dict):
        return dict(DEFAULT_QUANT_BITS)

    out: dict[str, float] = {}
    for key, value in raw_map.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = float(value)

    return out or dict(DEFAULT_QUANT_BITS)


def validate_catalog_files(
    data_dir: Path,
    *,
    quant_bits_expected: dict[str, float] | None = None,
    config_path: Path | None = None,
) -> dict[str, int]:
    expected_quant_bits = quant_bits_expected or _load_expected_quant_bits(config_path)

    errors: list[str] = []

    gpus_payload = _load_json(data_dir / "seed_gpus.json")
    models_payload = _load_json(data_dir / "seed_models.json")
    variants_payload = _load_json(data_dir / "seed_variants.json")

    gpu_items = _validate_envelope("seed_gpus.json", gpus_payload, errors)
    model_items = _validate_envelope("seed_models.json", models_payload, errors)
    variant_items = _validate_envelope("seed_variants.json", variants_payload, errors)

    versions = {
        value
        for value in [
            gpus_payload.get("catalog_version"),
            models_payload.get("catalog_version"),
            variants_payload.get("catalog_version"),
        ]
        if isinstance(value, str)
    }
    if len(versions) != 1:
        errors.append("catalog_version must match across all seed files")

    gpu_ids: set[str] = set()
    for idx, gpu in enumerate(gpu_items):
        label = f"seed_gpus.json[{idx}]"
        gpu_id = _expect_non_empty_string(gpu, "id", label, errors)
        if gpu_id is not None:
            if gpu_id in gpu_ids:
                errors.append(f"{label}: duplicate id {gpu_id!r}")
            gpu_ids.add(gpu_id)

        _expect_non_empty_string(gpu, "vendor", label, errors)
        _expect_non_empty_string(gpu, "name", label, errors)
        _expect_number(gpu, "vram_gb", label, errors)
        _expect_sources_list(gpu, label, errors)

    model_ids: set[str] = set()
    for idx, model in enumerate(model_items):
        label = f"seed_models.json[{idx}]"
        model_id = _expect_non_empty_string(model, "id", label, errors)
        if model_id is not None:
            if model_id in model_ids:
                errors.append(f"{label}: duplicate id {model_id!r}")
            model_ids.add(model_id)

        _expect_non_empty_string(model, "name", label, errors)
        _expect_non_empty_string(model, "family", label, errors)
        _expect_number(model, "params_b", label, errors)
        _expect_number(model, "kv_gb_per_1k_ctx", label, errors)
        _expect_sources_list(model, label, errors)

        if "license" not in model:
            errors.append(f"{label}: missing required field license")
        if "model_type" not in model:
            errors.append(f"{label}: missing required field model_type")

    allowed_quant_buckets = set(expected_quant_bits)

    variant_ids: set[str] = set()
    for idx, variant in enumerate(variant_items):
        label = f"seed_variants.json[{idx}]"
        variant_id = _expect_non_empty_string(variant, "id", label, errors)
        if variant_id is not None:
            if variant_id in variant_ids:
                errors.append(f"{label}: duplicate id {variant_id!r}")
            variant_ids.add(variant_id)

        model_id = _expect_non_empty_string(variant, "model_id", label, errors)
        if model_id is not None and model_id not in model_ids:
            errors.append(f"{label}: model_id {model_id!r} not found in seed_models.json")

        quant_bucket = _expect_non_empty_string(variant, "quant_bucket", label, errors)
        _expect_non_empty_string(variant, "quant_label", label, errors)

        bits_effective = _expect_number(variant, "bits_effective", label, errors)
        if quant_bucket is not None and quant_bucket not in allowed_quant_buckets:
            errors.append(
                f"{label}: quant_bucket {quant_bucket!r} is not allowed; "
                f"expected one of {sorted(allowed_quant_buckets)}"
            )

        if quant_bucket is not None and bits_effective is not None:
            expected = expected_quant_bits.get(quant_bucket)
            if expected is not None and abs(bits_effective - expected) > 1e-9:
                errors.append(
                    f"{label}: bits_effective={bits_effective} does not match expected "
                    f"{expected} for quant_bucket={quant_bucket}"
                )

        if "recommended" not in variant or not isinstance(variant.get("recommended"), bool):
            errors.append(f"{label}: recommended must be a boolean")

        _expect_sources_list(variant, label, errors)

    if errors:
        message = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Catalog validation failed:\n{message}")

    return {
        "gpus": len(gpu_items),
        "models": len(model_items),
        "variants": len(variant_items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VRAM Sherpa seed catalog JSON files.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Directory with seed JSON files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tools/catalog_sources.yaml"),
        help="Optional curated source config used to resolve expected quant bucket bits.",
    )
    args = parser.parse_args()

    try:
        summary = validate_catalog_files(args.data_dir, config_path=args.config)
    except ValueError as exc:
        print(exc)
        return 1

    print(
        "Catalog validation passed: "
        f"{summary['gpus']} GPUs, {summary['models']} models, {summary['variants']} variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
