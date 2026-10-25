"""R-075 / ADR-0007: the /how-it-works worked example is computed from the catalog."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session
from starlette.requests import Request

from vramsherpa import main
from vramsherpa.estimation import estimate_breakdown, estimate_variant
from vramsherpa.main import (
    ESTIMATION_POLICY_VERSION,
    WORKED_EXAMPLE_CONTEXT_TOKENS,
    WORKED_EXAMPLE_FALLBACK,
    WORKED_EXAMPLE_MODEL_ID,
    WORKED_EXAMPLE_QUANT_BUCKET,
    how_it_works,
)
from vramsherpa.models import Model, Variant

MODEL_LINK = "/models/model_llama_31_8b_instruct?context_tokens=2048"


def _page(app, session: Session) -> tuple[int, str]:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/how-it-works",
        "raw_path": b"/how-it-works",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "app": app,
        "router": app.router,
    }
    response = how_it_works(Request(scope), session=session)
    return response.status_code, response.body.decode("utf-8")


def _seed_items(data_dir: Path, name: str) -> list[dict]:
    return json.loads((data_dir / name).read_text(encoding="utf-8"))["items"]


def test_constants_and_policy_version() -> None:
    assert WORKED_EXAMPLE_MODEL_ID == "model_llama_31_8b_instruct"
    assert WORKED_EXAMPLE_QUANT_BUCKET == "Q4"
    assert WORKED_EXAMPLE_CONTEXT_TOKENS == 2048
    assert WORKED_EXAMPLE_FALLBACK == {
        "params_b": 8.0,
        "bits_effective": 4.5,
        "kv_gb_per_1k_ctx": 0.48,
    }
    assert ESTIMATION_POLICY_VERSION == "v0.1"


def test_worked_example_model_is_curated_in_seed_files(data_dir: Path) -> None:
    models = {item["id"]: item for item in _seed_items(data_dir, "seed_models.json")}
    assert WORKED_EXAMPLE_MODEL_ID in models
    assert not models[WORKED_EXAMPLE_MODEL_ID].get("discovered")
    variants = _seed_items(data_dir, "seed_variants.json")
    assert any(
        v["model_id"] == WORKED_EXAMPLE_MODEL_ID
        and v["quant_bucket"] == WORKED_EXAMPLE_QUANT_BUCKET
        for v in variants
    )


def test_worked_example_named_model_figures(app, seeded_session: Session) -> None:
    status, html = _page(app, seeded_session)
    assert status == 200
    assert "<h2>Worked example</h2>" in html
    assert "Llama 3.1 8B Instruct at Q4, with a 2048-token context" in html
    assert f'<a href="{MODEL_LINK}">See its model page</a>' in html
    for literal in ("4.95 GB", "0.98 GB", "1.40 GB", "7.33 GB"):
        assert literal in html
    for calc in (
        "8 &times; 4.5 &divide; 8 &times; 1.10 = 4.95",
        "0.48 &times; 2048 &divide; 1000 = 0.98304",
        "1.0 + 0.05 &times; 8 = 1.40",
        "4.95 + 0.98304 + 1.40 = 7.33304",
    ):
        assert calc in html
    for name, value in (
        ("params_b", "8"),
        ("bits_effective", "4.5"),
        ("kv_gb_per_1k_ctx", "0.48"),
        ("context_tokens", "2048"),
    ):
        assert f"<code>{name}</code> {value}</li>" in html


def test_worked_example_equals_card_figure(app, seeded_session: Session) -> None:
    model = seeded_session.get(Model, WORKED_EXAMPLE_MODEL_ID)
    variant = next(v for v in model.variants if v.quant_bucket == "Q4")
    kwargs = {
        "params_b": float(model.params_b),
        "bits_effective": float(variant.bits_effective),
        "kv_gb_per_1k_ctx": float(model.kv_gb_per_1k_ctx),
        "context_tokens": 2048,
        "available_vram_gb": 24.0,
    }
    breakdown = estimate_breakdown(**kwargs)
    card = estimate_variant(**kwargs).required_vram_gb
    assert breakdown.required_vram_gb == card
    _, html = _page(app, seeded_session)
    for figure in (
        breakdown.weights_gb,
        breakdown.kv_cache_gb,
        breakdown.runtime_overhead_gb,
        card,
    ):
        assert f"<strong>{figure:.2f} GB</strong>" in html


def test_worked_example_follows_catalog_values(app, seeded_session: Session) -> None:
    model = seeded_session.get(Model, WORKED_EXAMPLE_MODEL_ID)
    model.kv_gb_per_1k_ctx = 0.5
    seeded_session.flush()
    _, html = _page(app, seeded_session)
    expected = estimate_breakdown(
        params_b=float(model.params_b),
        bits_effective=4.5,
        kv_gb_per_1k_ctx=0.5,
        context_tokens=2048,
        available_vram_gb=24.0,
    )
    assert f"<strong>{expected.kv_cache_gb:.2f} GB</strong>" in html
    assert f"<strong>{expected.required_vram_gb:.2f} GB</strong>" in html
    assert "<code>kv_gb_per_1k_ctx</code> 0.5</li>" in html


def _assert_fallback(status: int, html: str) -> None:
    assert status == 200
    assert "<h2>Worked example</h2>" in html
    assert "an 8B model at Q4" in html
    for literal in ("4.95 GB", "0.98 GB", "1.40 GB", "7.33 GB"):
        assert literal in html
    assert "Llama" not in html
    assert "See its model page" not in html
    assert "/models/" not in html
    # every other section is kept
    for heading in ("Inputs", "Formula", "Reserve buffer", "Classification", "Limitations"):
        assert f"<h2>{heading}</h2>" in html
    assert "not a benchmark" in html


def test_fallback_when_catalog_unseeded(app, db_session: Session) -> None:
    _assert_fallback(*_page(app, db_session))


def test_fallback_when_model_missing(app, seeded_session: Session) -> None:
    model = seeded_session.get(Model, WORKED_EXAMPLE_MODEL_ID)
    for variant in list(model.variants):
        seeded_session.delete(variant)
    seeded_session.delete(model)
    seeded_session.flush()
    seeded_session.expire_all()
    _assert_fallback(*_page(app, seeded_session))


def test_fallback_when_no_q4_variant(app, seeded_session: Session) -> None:
    model = seeded_session.get(Model, WORKED_EXAMPLE_MODEL_ID)
    for variant in list(model.variants):
        if variant.quant_bucket == "Q4":
            seeded_session.delete(variant)
    seeded_session.flush()
    seeded_session.expire(model)
    assert (
        not seeded_session.query(Variant)
        .filter_by(model_id=WORKED_EXAMPLE_MODEL_ID, quant_bucket="Q4")
        .count()
    )
    _assert_fallback(*_page(app, seeded_session))


def test_estimation_module_untouched_by_this_task() -> None:
    assert main.estimate_breakdown is estimate_breakdown
