from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ingest_catalog import (
    _build_params_b_overrides,
    _detect_model_type,
    _discover_models_from_hf,
    _extract_params_b,
    _fetch_hf_model_list,
    _load_discovered_models,
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


def _hf_model(hf_id: str, downloads: int = 200_000, gated: bool | str = False) -> dict:
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


def test_discover_keeps_gated_models_and_flags_them() -> None:
    """Gating blocks weight downloads, not sizing, so gated repos are kept and flagged."""
    hf_response = [_hf_model("testorg/MyModel-7B", gated="manual")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover()

    assert len(result) == 1
    assert result[0]["gated"] is True
    assert "license acceptance required" in result[0]["notes"]


def test_discover_marks_ungated_models_as_not_gated() -> None:
    hf_response = [_hf_model("testorg/MyModel-7B")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover()

    assert result[0]["gated"] is False
    assert "license acceptance required" not in result[0]["notes"]


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


# ── _fetch_hf_model_list ──────────────────────────────────────────────────────


def test_fetch_hf_model_list_requests_gated_and_card_data() -> None:
    """The default list response omits `gated` and `cardData`, so they must be asked for."""

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

        def read(self) -> bytes:
            return b"[]"

    captured: list[str] = []

    def _fake_urlopen(req, timeout):  # noqa: ARG001 - signature mirrors urlopen
        captured.append(req.full_url)
        return _FakeResponse()

    with patch("ingest_catalog.request.urlopen", _fake_urlopen):
        _fetch_hf_model_list("testorg", timeout_seconds=8, retries=0, user_agent="test")

    assert len(captured) == 1
    url = captured[0]
    assert "expand%5B%5D=gated" in url
    assert "expand%5B%5D=cardData" in url
    assert "expand%5B%5D=downloads" in url


def test_discover_marks_models_as_discovered() -> None:
    hf_response = [_hf_model("testorg/MyModel-7B-Instruct")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover()
    assert len(result) == 1
    assert result[0]["discovered"] is True


# ── _load_discovered_models ───────────────────────────────────────────────────


def _write_seed_models(tmp_path: Path, items: list[dict]) -> None:
    payload = {"catalog_version": "2026-01-01", "items": items}
    (tmp_path / "seed_models.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_discovered_models_carries_previous_finds(tmp_path: Path) -> None:
    _write_seed_models(
        tmp_path,
        [
            {"id": "model_curated", "canonical_identifier": "myorg/Curated-7B"},
            {"id": "model_found", "canonical_identifier": "myorg/Found-7B", "discovered": True},
        ],
    )

    carried = _load_discovered_models(tmp_path, curated_models=[])

    assert [model["id"] for model in carried] == ["model_found"]
    # hf_repo is stripped before writing, so it has to be rebuilt for enrichment.
    assert carried[0]["hf_repo"] == "myorg/Found-7B"


def test_load_discovered_models_skips_entries_now_curated(tmp_path: Path) -> None:
    _write_seed_models(
        tmp_path,
        [
            {"id": "model_found", "canonical_identifier": "myorg/Found-7B", "discovered": True},
            {"id": "model_other", "canonical_identifier": "myorg/Other-7B", "discovered": True},
        ],
    )
    curated = [
        {
            "id": "model_hand_written",
            "canonical_identifier": "myorg/Found-7B",
            "hf_repo": "myorg/Found-7B",
        }
    ]

    carried = _load_discovered_models(tmp_path, curated_models=curated)

    assert [model["id"] for model in carried] == ["model_other"]


def test_load_discovered_models_skips_colliding_ids(tmp_path: Path) -> None:
    _write_seed_models(
        tmp_path,
        [{"id": "model_dupe", "canonical_identifier": "myorg/Dupe-7B", "discovered": True}],
    )
    curated = [{"id": "model_dupe", "canonical_identifier": "internal_dupe"}]

    assert _load_discovered_models(tmp_path, curated_models=curated) == []


def test_load_discovered_models_defaults_missing_gated(tmp_path: Path) -> None:
    """A catalog written before `gated` existed must stay valid after an upgrade."""
    _write_seed_models(
        tmp_path,
        [{"id": "model_old", "canonical_identifier": "myorg/Old-7B", "discovered": True}],
    )

    carried = _load_discovered_models(tmp_path, curated_models=[])

    assert len(carried) == 1
    assert carried[0]["gated"] is False


def test_load_discovered_models_no_seed_file(tmp_path: Path) -> None:
    assert _load_discovered_models(tmp_path, curated_models=[]) == []


# ── params_b overrides ────────────────────────────────────────────────────────


def test_extract_params_b_uses_override_by_full_id() -> None:
    assert _extract_params_b("microsoft/phi-4", {"microsoft/phi-4": 14.66}) == 14.66


def test_extract_params_b_override_lookup_is_case_insensitive() -> None:
    overrides = {"microsoft/phi-3.5-moe-instruct": 41.87}
    assert _extract_params_b("microsoft/Phi-3.5-MoE-instruct", overrides) == 41.87


def test_extract_params_b_override_takes_precedence_over_regex() -> None:
    # Llama-2-7b-hf parses to 7.0, but an explicit override wins.
    assert _extract_params_b("meta-llama/Llama-2-7b-hf", {"meta-llama/llama-2-7b-hf": 6.74}) == 6.74


def test_extract_params_b_falls_back_to_regex_when_not_overridden() -> None:
    assert _extract_params_b("Qwen/Qwen2.5-7B-Instruct", {"microsoft/phi-4": 14.66}) == 7.0


def test_discover_uses_params_b_overrides() -> None:
    hf_response = [_hf_model("testorg/SomeModel-mini")]
    with patch("ingest_catalog._fetch_hf_model_list", return_value=hf_response):
        result = _discover(params_b_overrides={"testorg/somemodel-mini": 3.82})

    assert len(result) == 1
    assert result[0]["params_b"] == 3.82
    # kv heuristic must follow the overridden parameter count
    assert result[0]["kv_gb_per_1k_ctx"] == round(0.06 * 3.82, 2)


def test_build_params_b_overrides_normalizes_and_validates() -> None:
    assert _build_params_b_overrides(None) == {}
    assert _build_params_b_overrides({"MicroSoft/Phi-4": 14.66}) == {"microsoft/phi-4": 14.66}

    with pytest.raises(ValueError, match="must be a mapping"):
        _build_params_b_overrides([("microsoft/phi-4", 14.66)])

    with pytest.raises(ValueError, match="params_b must be a number"):
        _build_params_b_overrides({"microsoft/phi-4": "14.66"})
