"""QA acceptance checks for R-074 (model detail page). Black-box, over HTTP.

Every figure is recomputed from the seed data with the R-009 to R-011 formulas, independently of
`estimation.py`. Phone layout, mockup match and tap targets are browser checks; see
.sdlc/qa/R-074/report.md.
"""

from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = {m["id"]: m for m in json.loads((DATA / "seed_models.json").read_text())["items"]}
VARIANTS = json.loads((DATA / "seed_variants.json").read_text())["items"]
HEADINGS = [
    "Variant",
    "Bits",
    "Required VRAM (GB)",
    "Available VRAM (GB)",
    "Margin (GB)",
    "Status",
    "Provenance",
]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _classify(required: float, available: float) -> str:
    reserve = max(0.75, 0.10 * available)
    if required <= available - reserve:
        return "Fits"
    if required <= available:
        return "Tight"
    return "Won't fit"


def _expected_rows(model_id: str, vram: float, context: int) -> list[list[str]]:
    model = MODELS[model_id]
    rows = []
    for v in sorted(
        (v for v in VARIANTS if v["model_id"] == model_id),
        key=lambda v: (v["quant_bucket"], v["quant_label"]),
    ):
        weights = model["params_b"] * v["bits_effective"] / 8 * 1.10
        required = (
            weights + model["kv_gb_per_1k_ctx"] * context / 1000 + 1.0 + 0.05 * model["params_b"]
        )
        rows.append(
            [
                f"{v['quant_label']} ({v['quant_bucket']})" + (" *" if v["recommended"] else ""),
                f"{v['bits_effective']:.1f}",
                f"{required:.2f}",
                f"{vram:.2f}",
                f"{vram - required:.2f}",
                _classify(required, vram),
            ]
        )
    return rows


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _rows(page: str) -> list[list[str]]:
    body = page.split("<tbody", 1)[1]
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        out.append([_text(c) for c in cells[:6]])
    return out


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_every_variant_row_shows_the_six_figures_and_the_classification(
    client: TestClient, model_id: str
) -> None:
    page = client.get(f"/models/{model_id}", params={"vram_gb": 12, "context_tokens": 4096}).text
    assert _rows(page) == _expected_rows(model_id, 12.0, 4096)


def test_figures_hold_at_other_hardware_and_context(client: TestClient) -> None:
    for model_id, vram, ctx in [
        ("model_llama_31_8b_instruct", 24, 2048),
        ("model_qwen25_7b_instruct", 24, 4096),
        ("model_qwen25_7b_instruct", 6, 8192),
    ]:
        page = client.get(f"/models/{model_id}", params={"vram_gb": vram, "context_tokens": ctx})
        assert _rows(page.text) == _expected_rows(model_id, float(vram), ctx)


def test_table_keeps_seven_headings_in_order_with_cell_labels_and_roles(
    client: TestClient,
) -> None:
    page = client.get("/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096").text
    assert [_text(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", page, re.S)] == HEADINGS
    assert "table-scroll" not in page
    first_row = re.findall(r"<tr[^>]*>(.*?)</tr>", page.split("<tbody", 1)[1], re.S)[0]
    assert re.findall(r'data-label="([^"]+)"', first_row) == HEADINGS
    assert page.count('role="cell"') == 7 * len(_expected_rows("model_qwen25_7b_instruct", 1, 1))


def test_provenance_follows_the_figures_and_classification_in_every_row(client: TestClient) -> None:
    page = client.get("/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096").text
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page.split("<tbody", 1)[1], re.S):
        labels = re.findall(r'data-label="([^"]+)"', tr)
        assert labels[-1] == "Provenance"
        assert tr.index('class="st"') < tr.index('class="prov"')
        assert tr.index('class="fig"') < tr.index('class="prov"')


def test_model_facts_kv_line_sources_and_404(client: TestClient) -> None:
    page = client.get("/models/model_qwen25_7b_instruct?vram_gb=24").text
    for label, value in [("Family", "qwen"), ("Params", "7.6B"), ("Type", "instruct")]:
        assert f"<dt>{label}</dt><dd>{value}</dd>" in page
    assert "KV cache estimate per 1k context: 0.460 GB" in page
    assert page.count('target="_blank" rel="noopener noreferrer"') >= 2
    assert 'class="sr-only">Recompute for other hardware<' in page
    assert 'class="sr-only">Variants<' in page
    missing = client.get("/models/nope")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Model not found"}


def test_gated_notice_and_marker_only_on_gated_models(client: TestClient) -> None:
    gated = client.get("/models/model_llama_31_8b_instruct").text
    assert "Weights are gated on Hugging Face: accept the model license" in gated
    assert ">gated<" in gated
    open_page = client.get("/models/model_qwen25_7b_instruct").text
    assert "Weights are gated" not in open_page
    assert ">gated<" not in open_page


def test_recompute_form_precedence_default_blank_and_invalid(client: TestClient) -> None:
    base = "/models/model_qwen25_7b_instruct"
    both = client.get(base, params={"gpu_id": "gpu_nvidia_rtx_4090_laptop_16gb", "vram_gb": 12})
    assert {r[3] for r in _rows(both.text)} == {"16.00"}
    assert "Choosing a GPU and entering VRAM together will prioritize the selected GPU." in (
        both.text
    )
    assert {r[3] for r in _rows(client.get(base).text)} == {"8.00"}
    blank = client.get(base, params={"gpu_id": "", "vram_gb": ""})
    assert blank.status_code == 200 and {r[3] for r in _rows(blank.text)} == {"8.00"}
    bad = client.get(base, params={"vram_gb": "abc"})
    assert bad.status_code == 200
    assert "Invalid value for vram_gb." in bad.text
    assert 'aria-invalid="true"' in bad.text
    assert 'inputmode="decimal"' in bad.text
    assert [o for o in re.findall(r"<option value=\"(\d+)\"", bad.text)] == ["2048", "4096", "8192"]
