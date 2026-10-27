"""QA acceptance checks for R-073 ("How this is estimated" breakdown). Black-box, over HTTP.

The five figures are recomputed here from the seed data with the formula R-020 and
`/how-it-works` state, independently of `estimation.py`, so a relabelling cannot change them.
Layout, loading state and mockup match are browser checks; see .sdlc/qa/R-073/report.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

DATA = Path(__file__).resolve().parents[1] / "data"
LABELS = [
    "Model weights",
    "Context memory (KV cache)",
    "Runtime overhead",
    "Total needed",
    "Safety reserve",
]
RAW = ["weights_gb", "kv_cache_gb", "runtime_overhead_gb", "reserve_gb", "required_vram_gb"]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _expected(variant_id: str, vram_gb: float, context_tokens: int) -> list[str]:
    variants = {v["id"]: v for v in json.loads((DATA / "seed_variants.json").read_text())["items"]}
    models = {m["id"]: m for m in json.loads((DATA / "seed_models.json").read_text())["items"]}
    variant = variants[variant_id]
    model = models[variant["model_id"]]
    weights = model["params_b"] * variant["bits_effective"] / 8 * 1.10
    kv = model["kv_gb_per_1k_ctx"] * context_tokens / 1000
    overhead = 1.0 + 0.05 * model["params_b"]
    reserve = max(0.75, 0.10 * vram_gb)
    return [f"{x:.2f} GB" for x in (weights, kv, overhead, weights + kv + overhead, reserve)]


@pytest.mark.parametrize(
    ("variant_id", "vram_gb", "context"),
    [
        ("variant_model_llama_31_8b_instruct_q4", 24, 4096),
        ("variant_model_llama_31_8b_instruct_q4", 8, 2048),
        ("variant_model_qwen25_7b_instruct_fp16", 24, 8192),
        ("variant_model_mistral_nemo_12b_instruct_2407_q8", 16, 4096),
        ("variant_model_deepseek_coder_v2_lite_instruct_q5", 3, 2048),
    ],
)
def test_breakdown_figures_match_the_formula_and_the_new_labels(
    client: TestClient, variant_id: str, vram_gb: int, context: int
) -> None:
    html = client.get(
        f"/results/why/{variant_id}", params={"vram_gb": vram_gb, "context_tokens": context}
    ).text
    labels = [re.sub(r"<[^>]+>.*", "", m).strip() for m in re.findall(r"<dt>(.*?)</dt>", html)]
    assert labels == LABELS
    assert re.findall(r'<dd class="num">([^<]+)</dd>', html) == _expected(
        variant_id, vram_gb, context
    )
    assert not any(raw in html for raw in RAW)
    assert html.index("Total needed") < html.index("Safety reserve")
    assert "Kept free, not part of the total" in html
