from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import yaml
from validate_catalog import validate_catalog_files

DEFAULT_CONFIG_PATH = Path("tools/catalog_sources.yaml")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_HF_TIMEOUT = 8
DEFAULT_HF_RETRIES = 2
DEFAULT_HF_USER_AGENT = "VRAMSherpaCatalogBot/1.0"


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    return slug.strip("_")


def _to_sources_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        out: list[str] = []
        for value in raw_value:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out.append(text)
        return out

    text = str(raw_value).strip()
    return [text] if text else []


def _to_optional_text(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text if text else None


def _require_number(raw_value: Any, field_name: str, item_label: str) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"{item_label}: {field_name} must be a number")
    return float(raw_value)


def _require_string(raw_value: Any, field_name: str, item_label: str) -> str:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{item_label}: {field_name} must be a non-empty string")
    return raw_value.strip()


def _read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return payload


def _fetch_hf_model_metadata(
    model_id: str,
    *,
    timeout_seconds: int,
    retries: int,
    user_agent: str,
) -> dict[str, Any] | None:
    encoded = parse.quote(model_id, safe="")
    url = f"https://huggingface.co/api/models/{encoded}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Unexpected HF response type for {model_id!r}")
                return payload
        except error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
            if 500 <= exc.code < 600 and attempt < retries:
                time.sleep(2**attempt)
                continue
            break
        except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            break

    if last_error is not None:
        raise RuntimeError(f"HF metadata fetch failed for {model_id!r}: {last_error}")
    return None


def _build_gpus(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("config.gpus must be a list")

    items: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_items):
        label = f"gpus[{idx}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: item must be an object")

        vendor = _require_string(raw.get("vendor"), "vendor", label)
        name = _require_string(raw.get("name"), "name", label)
        vram_gb = _require_number(raw.get("vram_gb"), "vram_gb", label)
        gpu_id = _to_optional_text(raw.get("id"))
        if gpu_id is None:
            gpu_id = f"gpu_{_slugify(vendor)}_{_slugify(name)}_{int(vram_gb)}gb"

        items.append(
            {
                "id": gpu_id,
                "vendor": vendor,
                "name": name,
                "vram_gb": vram_gb,
                "notes": _to_optional_text(raw.get("notes")),
                "sources": _to_sources_list(raw.get("sources")),
            }
        )

    return items


def _build_models(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("config.models must be a list")

    items: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_items):
        label = f"models[{idx}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: item must be an object")

        name = _require_string(raw.get("name"), "name", label)
        model_id = _to_optional_text(raw.get("id"))
        if model_id is None:
            model_id = f"model_{_slugify(name)}"

        family = _require_string(raw.get("family"), "family", label)
        params_b = _require_number(raw.get("params_b"), "params_b", label)
        kv_gb_per_1k_ctx = _require_number(raw.get("kv_gb_per_1k_ctx"), "kv_gb_per_1k_ctx", label)

        items.append(
            {
                "id": model_id,
                "name": name,
                "family": family,
                "params_b": params_b,
                "model_type": _to_optional_text(raw.get("model_type")),
                "license": _to_optional_text(raw.get("license")),
                "kv_gb_per_1k_ctx": kv_gb_per_1k_ctx,
                "notes": _to_optional_text(raw.get("notes")),
                "sources": _to_sources_list(raw.get("sources")),
                "hf_repo": _to_optional_text(raw.get("hf_repo")),
            }
        )

    return items


def _build_variants(
    raw_items: Any,
    *,
    quant_bits: dict[str, float],
) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("config.variants must be a list")

    items: list[dict[str, Any]] = []
    allowed_quant = set(quant_bits)

    for idx, raw in enumerate(raw_items):
        label = f"variants[{idx}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: item must be an object")

        model_id = _require_string(raw.get("model_id"), "model_id", label)
        quant_bucket = _require_string(raw.get("quant_bucket"), "quant_bucket", label)
        if quant_bucket not in allowed_quant:
            raise ValueError(
                f"{label}: quant_bucket={quant_bucket!r} not in {sorted(allowed_quant)}"
            )

        variant_id = _to_optional_text(raw.get("id"))
        if variant_id is None:
            variant_id = f"variant_{_slugify(model_id)}_{quant_bucket.lower()}"

        quant_label = _to_optional_text(raw.get("quant_label")) or quant_bucket
        bits_effective = raw.get("bits_effective")
        if bits_effective is None:
            bits_effective = quant_bits[quant_bucket]
        bits_effective = _require_number(bits_effective, "bits_effective", label)

        recommended = raw.get("recommended", False)
        if not isinstance(recommended, bool):
            raise ValueError(f"{label}: recommended must be a boolean")

        items.append(
            {
                "id": variant_id,
                "model_id": model_id,
                "quant_bucket": quant_bucket,
                "quant_label": quant_label,
                "bits_effective": bits_effective,
                "recommended": recommended,
                "notes": _to_optional_text(raw.get("notes")),
                "sources": _to_sources_list(raw.get("sources")),
            }
        )

    return items


def _enrich_models_from_hf(
    models: list[dict[str, Any]],
    *,
    enabled: bool,
    timeout_seconds: int,
    retries: int,
    user_agent: str,
    strict: bool,
) -> None:
    if not enabled:
        return

    for model in models:
        repo_id = model.get("hf_repo")
        if not isinstance(repo_id, str) or not repo_id:
            continue

        try:
            metadata = _fetch_hf_model_metadata(
                repo_id,
                timeout_seconds=timeout_seconds,
                retries=retries,
                user_agent=user_agent,
            )
        except RuntimeError as exc:
            if strict:
                raise
            print(f"Warning: {exc}")
            continue

        if not metadata:
            continue

        card_data = metadata.get("cardData")
        if isinstance(card_data, dict):
            license_value = card_data.get("license")
            if (
                isinstance(license_value, str)
                and license_value.strip()
                and not model.get("license")
            ):
                model["license"] = license_value.strip()

        hf_page = f"https://huggingface.co/{repo_id}"
        sources = model.get("sources")
        if isinstance(sources, list) and hf_page not in sources:
            sources.append(hf_page)


def _write_seed_file(path: Path, *, catalog_version: str, items: list[dict[str, Any]]) -> None:
    payload = {
        "catalog_version": catalog_version,
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _resolve_catalog_version(
    output_dir: Path,
    *,
    configured_version: str | None,
    gpus: list[dict[str, Any]],
    models: list[dict[str, Any]],
    variants: list[dict[str, Any]],
) -> str:
    if configured_version is not None:
        return configured_version

    today = datetime.now(UTC).date().isoformat()
    existing_paths = [
        output_dir / "seed_gpus.json",
        output_dir / "seed_models.json",
        output_dir / "seed_variants.json",
    ]
    if not all(path.exists() for path in existing_paths):
        return today

    try:
        existing_gpus = json.loads((output_dir / "seed_gpus.json").read_text(encoding="utf-8"))
        existing_models = json.loads((output_dir / "seed_models.json").read_text(encoding="utf-8"))
        existing_variants = json.loads(
            (output_dir / "seed_variants.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return today

    existing_items_match = (
        isinstance(existing_gpus, dict)
        and isinstance(existing_models, dict)
        and isinstance(existing_variants, dict)
        and existing_gpus.get("items") == gpus
        and existing_models.get("items") == models
        and existing_variants.get("items") == variants
    )
    if not existing_items_match:
        return today

    existing_version = existing_gpus.get("catalog_version")
    if isinstance(existing_version, str) and existing_version:
        return existing_version
    return today


def _summarize(
    gpus: list[dict[str, Any]], models: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> dict[str, int]:
    return {
        "gpus": len(gpus),
        "models": len(models),
        "variants": len(variants),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate seed catalog JSON files from curated sources."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to curated source config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where seed JSON files are written.",
    )
    parser.add_argument(
        "--catalog-version",
        type=str,
        default=None,
        help="Optional override for catalog_version (YYYY-MM-DD). Default is current UTC date.",
    )
    parser.add_argument(
        "--with-hf-metadata",
        action="store_true",
        help="Optionally enrich configured models with metadata from the Hugging Face API.",
    )
    parser.add_argument(
        "--hf-timeout-seconds",
        type=int,
        default=DEFAULT_HF_TIMEOUT,
        help="Hugging Face API timeout in seconds.",
    )
    parser.add_argument(
        "--hf-retries",
        type=int,
        default=DEFAULT_HF_RETRIES,
        help="Retry count for Hugging Face API calls.",
    )
    parser.add_argument(
        "--hf-user-agent",
        type=str,
        default=DEFAULT_HF_USER_AGENT,
        help="User-Agent header for Hugging Face API requests.",
    )
    parser.add_argument(
        "--strict-hf",
        action="store_true",
        help="Fail ingest if any Hugging Face metadata call fails.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path to write summary JSON (counts).",
    )
    args = parser.parse_args()

    config = _read_config(args.config)

    catalog_config = config.get("catalog")
    if not isinstance(catalog_config, dict):
        catalog_config = {}

    quant_bits = catalog_config.get("quant_bucket_bits_effective")
    if not isinstance(quant_bits, dict):
        raise ValueError("catalog.quant_bucket_bits_effective must be a mapping")

    normalized_quant_bits: dict[str, float] = {}
    for key, value in quant_bits.items():
        if not isinstance(key, str):
            raise ValueError("catalog.quant_bucket_bits_effective keys must be strings")
        normalized_quant_bits[key] = _require_number(
            value, "quant_bucket_bits_effective", "catalog"
        )

    gpus = _build_gpus(config.get("gpus"))
    models = _build_models(config.get("models"))
    variants = _build_variants(config.get("variants"), quant_bits=normalized_quant_bits)

    hf_config = catalog_config.get("huggingface")
    if not isinstance(hf_config, dict):
        hf_config = {}

    enabled_by_config = bool(hf_config.get("enabled", False))
    hf_timeout = int(hf_config.get("timeout_seconds", args.hf_timeout_seconds))
    hf_retries = int(hf_config.get("retries", args.hf_retries))
    hf_user_agent = str(hf_config.get("user_agent") or args.hf_user_agent)

    _enrich_models_from_hf(
        models,
        enabled=(enabled_by_config or args.with_hf_metadata),
        timeout_seconds=hf_timeout,
        retries=hf_retries,
        user_agent=hf_user_agent,
        strict=args.strict_hf,
    )

    for model in models:
        model.pop("hf_repo", None)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_version = _resolve_catalog_version(
        output_dir,
        configured_version=args.catalog_version,
        gpus=gpus,
        models=models,
        variants=variants,
    )

    _write_seed_file(output_dir / "seed_gpus.json", catalog_version=catalog_version, items=gpus)
    _write_seed_file(output_dir / "seed_models.json", catalog_version=catalog_version, items=models)
    _write_seed_file(
        output_dir / "seed_variants.json", catalog_version=catalog_version, items=variants
    )

    summary = validate_catalog_files(
        output_dir,
        quant_bits_expected=normalized_quant_bits,
        config_path=args.config,
    )

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(
        f"Catalog refresh completed for {catalog_version}: "
        f"{summary['gpus']} GPUs, {summary['models']} models, {summary['variants']} variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
