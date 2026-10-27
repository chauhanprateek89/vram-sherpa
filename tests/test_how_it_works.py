"""R-075 / ADR-0007: the /how-it-works worked example is computed from the catalog."""

from __future__ import annotations

import json
import re
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
        assert f'<div><dt>{name}</dt><dd class="num">{value}</dd></div>' in html


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
        assert f'<span class="out num">{figure:.2f} GB</span>' in html


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
    assert f'<span class="out num">{expected.kv_cache_gb:.2f} GB</span>' in html
    assert f'<span class="out num">{expected.required_vram_gb:.2f} GB</span>' in html
    assert '<div><dt>kv_gb_per_1k_ctx</dt><dd class="num">0.5</dd></div>' in html


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


SECTIONS = [
    "Inputs",
    "Formula",
    "Reserve buffer",
    "Classification",
    "Worked example",
    "Limitations",
]
FORMULAS = (
    "weights_gb = params_b * bits_effective / 8 * 1.10",
    "kv_cache_gb = kv_gb_per_1k_ctx * (context_tokens / 1000)",
    "runtime_overhead_gb = 1.0 + 0.05 * params_b",
    "required_vram_gb = weights_gb + kv_cache_gb + runtime_overhead_gb",
    "reserve_gb = max(0.75, 0.10 * available_vram_gb)",
)
RULES = (
    "required_vram_gb &lt;= available_vram_gb - reserve_gb",
    "available_vram_gb - reserve_gb &lt; required_vram_gb &lt;= available_vram_gb",
    "required_vram_gb &gt; available_vram_gb",
)
TERMS = (
    "params_b",
    "bits_effective",
    "kv_gb_per_1k_ctx",
    "context_tokens",
    "weights_gb",
    "kv_cache_gb",
    "runtime_overhead_gb",
    "required_vram_gb",
    "available_vram_gb",
    "reserve_gb",
)
EXPLANATIONS = (
    "The VRAM you have, from the GPU you pick or the number you type.",
    "How much text the model keeps in view at once: 2048, 4096 or 8192 tokens, shown as "
    "2k, 4k and 8k.",
    "Model size in billions of parameters. A bigger model has more weights to store.",
    "Bits stored per parameter after quantisation: Q4 is 4.5, Q5 is 5.5, Q8 is 8.5, FP16 is "
    "16. Fewer bits, less memory.",
    "Memory the model's context (its KV cache) takes for every 1,000 tokens, in GB. "
    "Each model page shows this value.",
    "Memory for the model itself. Parameters times bits per parameter, divided by 8, gives "
    "gigabytes; the 1.10 adds 10% on top.",
    "Memory for the context window you chose. It grows with the context length.",
    "Memory the runtime needs besides the model: a flat 1.0 GB plus 0.05 GB for every billion "
    "parameters.",
    "Everything the variant needs: weights, KV cache and runtime overhead added up.",
    "Free space to leave as a safety margin: 10% of your VRAM, never less than 0.75 GB.",
)
SOURCE_LINES = (
    "From GPU selection or manual VRAM input",
    "From presets (2048 default, 4096, 8192)",
    "Variant and model metadata",
)


def test_headings_and_policy_version(app, seeded_session: Session, monkeypatch) -> None:
    # a made-up version proves the heading and footer read it rather than hard-code "v0.1"
    monkeypatch.setattr(main, "ESTIMATION_POLICY_VERSION", "v9.9")
    _, html = _page(app, seeded_session)
    assert html.count("<h1>") == 1
    assert "<h1>How VRAM Sherpa estimates fit (Policy v9.9)</h1>" in html
    assert "Estimation policy: v9.9</span>" in html
    assert re.findall(r"<h2>(.*?)</h2>", html) == SECTIONS


def test_formulas_rules_intro_and_limitations_word_for_word(app, seeded_session: Session) -> None:
    _, html = _page(app, seeded_session)
    for text in FORMULAS:
        assert f'<code class="formula">{text}</code>' in html
    for text in RULES:
        assert f"<code>{text}</code>" in html
    assert (
        "VRAM Sherpa estimates whether a model variant can fit on your available VRAM for "
        "local inference. It is a planning estimate, not a benchmark."
    ) in html
    assert (
        "Real outcomes vary by runtime, driver stack, memory fragmentation, and CPU/GPU "
        "offload behavior. This tool is not a benchmark and does not guarantee performance."
    ) in html
    for constant in ("1.10", "1.0", "0.05", "0.75", "0.10"):
        assert constant in html


def test_every_term_is_explained_in_plain_language(app, seeded_session: Session) -> None:
    _, html = _page(app, seeded_session)
    for term in TERMS:
        assert f'<code class="name">{term}</code>' in html
    for line in EXPLANATIONS:
        assert f"<p>{line}</p>" in html
    for line in SOURCE_LINES:
        assert f'<p class="src">{line}</p>' in html


def test_reserve_gauge_reads_reserve_gb(app, seeded_session: Session, monkeypatch) -> None:
    monkeypatch.setattr(main, "reserve_gb", lambda available: 2.5)
    _, html = _page(app, seeded_session)
    assert "The reserve is 2.5 GB. Up to 7.5 GB Fits, up to 10.0 GB is Tight" in html
    assert 'class="bar bar-fits" style="width: 75%"' in html
    assert 'class="resv" style="left: 75%; width: 25%"' in html


def test_reserve_examples_and_gauge(app, seeded_session: Session) -> None:
    _, html = _page(app, seeded_session)
    assert "With 4 GB the reserve is 0.75 GB. With 16 GB it is 1.6 GB." in html
    assert "Example with 10 GB of VRAM" in html
    assert (
        "The reserve is 1.0 GB. Up to 9.0 GB Fits, up to 10.0 GB is Tight, and anything more "
        "Won't fit."
    ) in html
    # widths come from reserve_gb(10): 9.0 GB of 10 is 90%, the 1.0 GB reserve the last 10%
    assert 'class="bar bar-tight" style="width: 100%"' in html
    assert 'class="bar bar-fits" style="width: 90%"' in html
    assert 'class="resv" style="left: 90%; width: 10%"' in html
    assert main._reserve_example()["reserve_gb"] == "1.0"


def test_classification_tiles_use_the_verdict_macro(app, seeded_session: Session) -> None:
    _, html = _page(app, seeded_session)
    for key, label in (("fits", "Fits"), ("tight", "Tight"), ("wont", "Won&#39;t fit")):
        assert (
            f'<span class="state state-{key}"><svg width="18" height="18" aria-hidden="true">'
            f'<use href="#i-{key}"/></svg>{label}</span>'
        ) in html


def test_assets_bumped_for_this_page() -> None:
    assert main.ASSET_VERSION > "20261025p"
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)


def _css() -> str:
    return (Path(main.__file__).parent / "static" / "css" / "styles.css").read_text("utf-8")


def _rule(css: str, selector: str) -> str:
    """The declaration block of the first rule whose selector list has exactly `selector`."""
    pattern = rf"(?m)^(?:[^{{}}\n]*,\s*)?{re.escape(selector)}(?:\s*,[^{{}}\n]*)? \{{([^}}]*)\}}"
    match = re.search(pattern, css)
    assert match, f"no rule for {selector}"
    return match.group(1)


def test_formulas_wrap_at_phone_width() -> None:
    css = _css()
    assert "overflow-wrap: anywhere" in _rule(css, ".doc code")
    assert "overflow-wrap: anywhere" in _rule(css, ".steps .calc")
    assert "white-space: nowrap" not in _rule(css, ".term code.formula")
    assert "white-space: nowrap" not in _rule(css, ".rule code")


def test_step_circles_stay_30px_and_links_use_accent() -> None:
    css = _css()
    circle = _rule(css, ".steps li::before")
    # the 1px border must not add 2px: border-box here, or a reset that reaches pseudo-elements
    reset = re.search(r"(?m)^\*, \*::before, \*::after \{ box-sizing: border-box; \}", css)
    assert "box-sizing: border-box" in circle or reset
    assert "width: 30px" in circle and "height: 30px" in circle
    # the page body link takes the mockup's accent, and no later global hover recolours it
    assert "color: var(--accent)" in _rule(css, ".doc a")
    assert "a:hover {" not in css or css.index(".doc a {") > css.index("a:hover {")


def test_estimation_module_untouched_by_this_task() -> None:
    assert main.estimate_breakdown is estimate_breakdown
