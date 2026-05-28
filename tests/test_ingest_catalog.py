from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ingest_catalog import (
    _detect_model_type,
    _discover_models_from_hf,
    _extract_params_b,
    _load_known_hf_ids,
)

# ── _extract_params_b ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", 8.0),
        ("meta-llama/Llama-3.1-70B-Instruct", 70.0),
        ("meta-llama/Llama-3.2-1B-Instruct", 1.0),
        ("Qwen/Qwen2.5-72B-Instruct", 72.0),
        ("Qwen/Qwen3-0.6B", 0.6),
        ("google/gemma-2-9b-it", 9.0),
        ("google/gemma-2-27b-it", 27.0),
        ("mistralai/Mistral-7B-Instruct-v0.3", 7.0),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 32.0),
        ("01-ai/Yi-34B-Chat", 34.0),
    ],
)
def test_extract_params_b_known_models(model_id: str, expected: float) -> None:
    assert _extract_params_b(model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/Phi-4",  # no B suffix
        "microsoft/Phi-3.5-mini-instruct",  # "mini" has no B count
        "mistralai/Mistral-Nemo-Instruct-2407",  # date suffix, no B count
        "some-org/NoSizeInName",
    ],
)
def test_extract_params_b_unresolvable(model_id: str) -> None:
    assert _extract_params_b(model_id) is None


# ── _detect_model_type ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "instruct"),
        ("Qwen/Qwen2.5-7B-Instruct", "instruct"),
        ("google/gemma-2-9b-it", "instruct"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "instruct"),
        ("meta-llama/Llama-2-7b-chat-hf", "instruct"),
        ("meta-llama/Llama-3.1-8B", "base"),
        ("Qwen/Qwen3-8B", "base"),
        ("google/gemma-2-9b", "base"),
        ("mistralai/Mistral-7B-v0.1", "base"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "base"),
    ],
)
def test_detect_model_type(model_id: str, expected: str) -> None:
    assert _detect_model_type(model_id) == expected


# ── helpers ───────────────────────────────────────────────────────────────────

_MOCK_ORG = [{"id": "testorg", "family": "testfam"}]


def _hf_model(hf_id: str, downloads: int = 200_000, gated: bool = False) -> dict:
    return {
        "id": hf_id,
        "downloads": downloads,
        "gated": gated,
        "cardData": {"license": "apache-2.0"},
        "pipeline_tag": "text-generation",
    }


def _discover(**kwargs) -> list:
    defaults = dict(
        tracked_orgs=_MOCK_ORG,
        known_hf_ids=set(),
        known_model_ids=set(),
        min_downloads=1_000,
        max_per_org=20,
        exclude_name_patterns=[],
        timeout_seconds=8,
        retries=0,
        user_agent="test",
    )
    defaults.update(kwargs)
    return _discover_models_from_hf(**defaults)


# ── _discover_models_from_hf ──────────────────────────────────────────────────


def test_discover_basic_model() -> None:
    hf_response = [_hf_model("testorg/MyModel-7B-Instruct")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover()

    assert len(result) == 1
    m = result[0]
    assert m["canonical_identifier"] == "testorg/MyModel-7B-Instruct"
    assert m["params_b"] == 7.0
    assert m["model_type"] == "instruct"
    assert m["family"] == "testfam"
    assert m["kv_gb_per_1k_ctx"] == round(0.06 * 7.0, 2)
    assert m["license"] == "apache-2.0"
    assert "https://huggingface.co/testorg/MyModel-7B-Instruct" in m["sources"]


def test_discover_skips_known_hf_id() -> None:
    hf_response = [_hf_model("testorg/MyModel-7B-Instruct")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover(known_hf_ids={"testorg/MyModel-7B-Instruct"})
    assert result == []


def test_discover_skips_below_min_downloads() -> None:
    hf_response = [_hf_model("testorg/MyModel-7B", downloads=100)]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover(min_downloads=50_000)
    assert result == []


def test_discover_skips_gated_models() -> None:
    hf_response = [_hf_model("testorg/MyModel-7B", gated=True)]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover()
    assert result == []


def test_discover_skips_unresolvable_params_b() -> None:
    hf_response = [_hf_model("testorg/SomeModel-mini")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover()
    assert result == []


def test_discover_skips_excluded_name_patterns() -> None:
    hf_response = [
        _hf_model("testorg/MyModel-7B-AWQ"),
        _hf_model("testorg/MyGuard-8B"),
        _hf_model("testorg/GoodModel-13B"),
    ]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover(exclude_name_patterns=["-AWQ", "Guard"])

    assert len(result) == 1
    assert result[0]["canonical_identifier"] == "testorg/GoodModel-13B"


def test_discover_respects_max_per_org() -> None:
    hf_response = [_hf_model(f"testorg/Model-{i}B") for i in range(1, 11)]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover(max_per_org=3)
    assert len(result) == 3


def test_discover_applies_org_name_filter() -> None:
    org = [{"id": "google", "family": "gemma", "model_name_contains": "gemma"}]
    hf_response = [
        _hf_model("google/gemma-2-9b-it"),
        _hf_model("google/bert-large-uncased"),
    ]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover(tracked_orgs=org)

    assert len(result) == 1
    assert "gemma" in result[0]["canonical_identifier"]


def test_discover_deduplicates_within_run() -> None:
    """Two orgs returning the same model ID should only produce one entry."""
    orgs = [
        {"id": "org-a", "family": "llama"},
        {"id": "org-b", "family": "llama"},
    ]
    hf_response = [_hf_model("shared-org/SharedModel-7B")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover_models_from_hf(
            tracked_orgs=orgs,
            known_hf_ids=set(),
            known_model_ids=set(),
            min_downloads=1_000,
            max_per_org=20,
            exclude_name_patterns=[],
            timeout_seconds=8,
            retries=0,
            user_agent="test",
        )

    # Both orgs return same model_id after slugify → second org skips it
    model_ids = [m["id"] for m in result]
    assert len(model_ids) == len(set(model_ids))


def test_discover_handles_fetch_failure_gracefully() -> None:
    with patch("ingest_catalog._fetch_hf_model_list", return_value=None):
        result = _discover()
    assert result == []


# ── _load_known_hf_ids ────────────────────────────────────────────────────────


def test_load_known_hf_ids_from_yaml_models(tmp_path: Path) -> None:
    yaml_models = [
        {
            "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
            "canonical_identifier": "meta-llama/Llama-3.1-8B-Instruct",
        },
        {"hf_repo": None, "canonical_identifier": "internal_id_no_slash"},
    ]
    known = _load_known_hf_ids(tmp_path, yaml_models)
    assert "meta-llama/Llama-3.1-8B-Instruct" in known
    assert "internal_id_no_slash" not in known  # no slash → not a HF id


def test_load_known_hf_ids_from_existing_seed_file(tmp_path: Path) -> None:
    seed = {
        "catalog_version": "2026-01-01",
        "items": [
            {"id": "model_foo", "canonical_identifier": "myorg/MyModel-7B"},
            {"id": "model_bar", "canonical_identifier": "internal_id_no_slash"},
        ],
    }
    (tmp_path / "seed_models.json").write_text(json.dumps(seed), encoding="utf-8")

    known = _load_known_hf_ids(tmp_path, [])
    assert "myorg/MyModel-7B" in known
    assert "internal_id_no_slash" not in known


def test_load_known_hf_ids_no_seed_file(tmp_path: Path) -> None:
    known = _load_known_hf_ids(tmp_path, [])
    assert known == set()
