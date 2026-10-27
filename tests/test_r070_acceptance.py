"""QA acceptance checks for R-070 (phone layout). Black-box, over HTTP.

Pixel measures (scroll width, tap targets, clipping, font sizes) are taken in a browser; see
.sdlc/qa/R-070/report.md. These check what the markup can prove: numeric keyboards.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

NUMERIC = {"decimal", "numeric"}


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _input_tag(html: str, field_id: str) -> str:
    match = re.search(rf"<input\b[^>]*\bid=\"{field_id}\"[^>]*>", html, re.S)
    assert match, f"no input #{field_id}"
    return match.group(0)


@pytest.mark.parametrize("field_id", ["vram_gb", "min_params_b", "max_params_b"])
def test_home_number_fields_ask_for_a_numeric_keyboard(client: TestClient, field_id: str) -> None:
    tag = _input_tag(client.get("/").text, field_id)
    assert 'type="number"' in tag
    mode = re.search(r'inputmode="([^"]+)"', tag)
    assert mode and mode.group(1) in NUMERIC


def test_model_detail_manual_vram_asks_for_a_numeric_keyboard(client: TestClient) -> None:
    html = client.get("/models/model_qwen25_7b_instruct?vram_gb=24").text
    tag = _input_tag(html, "vram_gb")
    mode = re.search(r'inputmode="([^"]+)"', tag)
    assert mode and mode.group(1) in NUMERIC


def test_results_page_keeps_numeric_keyboard_fields(client: TestClient) -> None:
    html = client.get("/results?vram_gb=24&context_tokens=4096").text
    for field_id in ("vram_gb", "min_params_b", "max_params_b"):
        assert 'inputmode="decimal"' in _input_tag(html, field_id)
