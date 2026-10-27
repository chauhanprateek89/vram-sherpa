"""QA acceptance checks for R-075 (/how-it-works). Black-box, over HTTP.

The worked example is recomputed here from the seed data and the R-009 formulas, independently of
`estimation.py`. Phone wrapping, theme rendering and mockup match are browser checks; see
.sdlc/qa/R-075/report.md.
"""

from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from vramsherpa import database

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL_ID = "model_llama_31_8b_instruct"
FORMULAS = [
    "weights_gb = params_b * bits_effective / 8 * 1.10",
    "kv_cache_gb = kv_gb_per_1k_ctx * (context_tokens / 1000)",
    "runtime_overhead_gb = 1.0 + 0.05 * params_b",
    "required_vram_gb = weights_gb + kv_cache_gb + runtime_overhead_gb",
    "reserve_gb = max(0.75, 0.10 * available_vram_gb)",
]
RULES = [
    "required_vram_gb &lt;= available_vram_gb - reserve_gb",
    "available_vram_gb - reserve_gb &lt; required_vram_gb &lt;= available_vram_gb",
    "required_vram_gb &gt; available_vram_gb",
]
LIMITATIONS = (
    "Real outcomes vary by runtime, driver stack, memory fragmentation, and CPU/GPU offload "
    "behavior. This tool is not a benchmark and does not guarantee performance."
)
TERMS = [
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
]
STEPS = ["4.95 GB", "0.98 GB", "1.40 GB", "7.33 GB"]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _example(page: str) -> dict[str, object]:
    section = page.split("<h2>Worked example</h2>", 1)[1].split("<h2>Limitations</h2>", 1)[0]
    inputs = re.findall(r"<dt>(.*?)</dt><dd[^>]*>(.*?)</dd>", section)
    outs = re.findall(r'<span class="out num">(.*?)</span>', section)
    return {"inputs": dict(inputs), "steps": outs, "lead": _text(section.split("</p>", 1)[0])}


def test_page_keeps_formulas_rules_constants_limitations_and_one_h1_six_h2(
    client: TestClient,
) -> None:
    page = client.get("/how-it-works").text
    assert len(re.findall(r"<h1[ >]", page)) == 1
    assert re.findall(r"<h2>(.*?)</h2>", page) == [
        "Inputs",
        "Formula",
        "Reserve buffer",
        "Classification",
        "Worked example",
        "Limitations",
    ]
    for formula in FORMULAS + RULES:
        assert formula in page
    assert LIMITATIONS in _text(page)
    for constant in ("1.10", "1.0", "0.05", "0.75", "0.10"):
        assert constant in page
    assert "With 4 GB the reserve is 0.75 GB. With 16 GB it is 1.6 GB." in _text(page)


def test_policy_version_equals_the_footer_and_is_v0_1(client: TestClient) -> None:
    page = client.get("/how-it-works").text
    heading = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    assert heading and _text(heading.group(1)) == "How VRAM Sherpa estimates fit (Policy v0.1)"
    footer = re.search(r"Estimation policy:\s*([^<\s]+)", page)
    assert footer and footer.group(1) == "v0.1"


def test_every_term_is_explained_in_plain_language(client: TestClient) -> None:
    page = client.get("/how-it-works").text
    for term in TERMS:
        block = re.search(rf'<code class="name">{term}</code>\s*<p>(.+?)</p>', page, re.S)
        assert block, term
        explanation = _text(block.group(1))
        assert len(explanation.split()) >= 5, term
        assert term not in explanation


def test_worked_example_names_the_catalog_model_and_matches_its_card_and_row(
    client: TestClient,
) -> None:
    page = client.get("/how-it-works").text
    ex = _example(page)
    assert "Llama 3.1 8B Instruct at Q4, with a 2048-token context" in ex["lead"]
    assert ex["inputs"] == {
        "params_b": "8",
        "bits_effective": "4.5",
        "kv_gb_per_1k_ctx": "0.48",
        "context_tokens": "2048",
    }
    assert ex["steps"] == STEPS
    assert 'href="/models/model_llama_31_8b_instruct?context_tokens=2048"' in page
    detail = client.get(f"/models/{MODEL_ID}", params={"vram_gb": 24, "context_tokens": 2048}).text
    q4 = next(r for r in re.findall(r"<tr[^>]*>(.*?)</tr>", detail, re.S) if "(Q4)" in r)
    assert re.search(r'data-label="Required VRAM \(GB\)">7\.33<', q4)
    why = client.get(
        f"/results/why/variant_{MODEL_ID}_q4", params={"vram_gb": 24, "context_tokens": 2048}
    ).text
    assert re.findall(r'<dd class="num">([^<]+)</dd>', why)[:4] == [
        "4.95 GB",
        "0.98 GB",
        "1.40 GB",
        "7.33 GB",
    ]
    results = client.get(
        "/results", params={"vram_gb": 24, "context_tokens": 2048, "family": "llama"}
    ).text
    card = results.split("Llama 3.1 8B Instruct", 1)[1].split("</article>", 1)[0]
    assert "7.33 GB" in _text(card)


def test_example_follows_the_catalog_values(client: TestClient) -> None:
    with database.SessionLocal() as session:
        session.execute(
            text("update models set kv_gb_per_1k_ctx = 0.5 where id = :i"), {"i": MODEL_ID}
        )
        session.execute(
            text("update variants set bits_effective = 5.5 where id = :i"),
            {"i": f"variant_{MODEL_ID}_q4"},
        )
        session.commit()
    ex = _example(client.get("/how-it-works").text)
    assert ex["inputs"]["kv_gb_per_1k_ctx"] == "0.5"
    assert ex["inputs"]["bits_effective"] == "5.5"
    weights = 8 * 5.5 / 8 * 1.10
    required = weights + 0.5 * 2048 / 1000 + 1.4
    assert ex["steps"] == [f"{weights:.2f} GB", "1.02 GB", "1.40 GB", f"{required:.2f} GB"]
    detail = client.get(f"/models/{MODEL_ID}", params={"vram_gb": 24, "context_tokens": 2048}).text
    q4 = next(r for r in re.findall(r"<tr[^>]*>(.*?)</tr>", detail, re.S) if "(Q4)" in r)
    assert f'data-label="Required VRAM (GB)">{required:.2f}<' in q4


@pytest.mark.parametrize("variant", ["no_model", "no_q4", "unseeded"])
def test_missing_model_or_q4_gives_the_unnamed_fallback(client: TestClient, variant: str) -> None:
    names = [m["name"] for m in json.loads((DATA / "seed_models.json").read_text())["items"]]
    with database.SessionLocal() as session:
        if variant == "no_model":
            session.execute(text("delete from variants where model_id = :i"), {"i": MODEL_ID})
            session.execute(text("delete from models where id = :i"), {"i": MODEL_ID})
        elif variant == "no_q4":
            session.execute(
                text("delete from variants where model_id = :i and quant_bucket = 'Q4'"),
                {"i": MODEL_ID},
            )
        else:
            session.execute(text("delete from variants"))
            session.execute(text("delete from models"))
        session.commit()
    response = client.get("/how-it-works")
    assert response.status_code == 200
    page = response.text
    assert len(re.findall(r"<h2>", page)) == 6
    for formula in FORMULAS + RULES:
        assert formula in page
    assert LIMITATIONS in _text(page)
    ex = _example(page)
    assert ex["lead"] == "An example for an 8B model at Q4, with a 2048-token context."
    assert ex["inputs"] == {
        "params_b": "8",
        "bits_effective": "4.5",
        "kv_gb_per_1k_ctx": "0.48",
        "context_tokens": "2048",
    }
    assert ex["steps"] == STEPS
    assert not any(name in page for name in names)
    assert "Llama" not in page and 'href="/models/' not in page
