"""R-083 / R-003: /results reads a box number in gpu_search as VRAM (ADR-0011 points 2 to 4)."""

from __future__ import annotations

import html as htmllib
import re
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

HX = {"HX-Request": "true"}
BOX_ATTR = re.compile(r'data-box-vram="[^"]*"')


def _norm(markup: str) -> str:
    return " ".join(BOX_ATTR.sub("", markup).split())


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _get(client: TestClient, query: str, *, htmx: bool = False) -> str:
    path = "/results/partials/content" if htmx else "/results"
    response = client.get(f"{path}?{query}", headers=HX if htmx else {})
    assert response.status_code == 200
    return htmllib.unescape(response.text)


def _both(client: TestClient, search: str):
    query = f"gpu_search={quote(search)}"
    return [_get(client, query), _get(client, query, htmx=True)]


@pytest.mark.parametrize(
    ("search", "label", "number"),
    [
        ("12 GB", "Manual VRAM (12 GB)", "12"),
        ("12.5gb", "Manual VRAM (12.5 GB)", "12.5"),
        ("8GB", "Manual VRAM (8 GB)", "8"),
        ("24", "Manual VRAM (24 GB)", "24"),
    ],
)
def test_box_number_is_vram(client, search, label, number):
    pages = _both(client, search)
    for page in pages:
        assert label in page
        assert "RTX 4060 Ti" not in page.split("Hardware:")[1].split("</span>")[0]
        assert "matched multiple entries" not in page
        assert "was not found" not in page
        assert f'data-box-vram="{number}"' in page
    assert 'name="gpu_search"' in pages[0]
    full = pages[0]
    assert f'value="{number}"' in full[full.index('id="vram_gb"') - 200 :]
    gpu_search_tag = re.search(r"<input[^>]*id=\"gpu_search\"[^>]*>", full).group(0)
    assert 'value="' not in gpu_search_tag or 'value=""' in gpu_search_tag


def test_box_number_page_equals_vram_gb_page(client):
    box = _get(client, "gpu_search=12%20GB&family=qwen")
    plain = _get(client, "vram_gb=12&family=qwen")
    assert _norm(box) == _norm(plain)
    assert "data-box-vram" not in plain
    assert "gpu_search=" not in box.split('id="results-shell"')[1]


@pytest.mark.parametrize("search", ["0", "0 GB", "0.0"])
def test_zero_matches_vram_zero(client, search):
    reference = _get(client, "vram_gb=0")
    assert "8 GB" in reference
    for page in _both(client, search):
        assert "was not found" not in page
        assert "matched multiple entries" not in page
        assert f'data-box-vram="{search.split()[0]}"' in page
    full = _norm(_both(client, search)[0])
    assert _norm(full) == _norm(reference).replace('value="0"', f'value="{search.split()[0]}"')


@pytest.mark.parametrize("search", ["nvidia geforce rtx 4090 (24.0 gb)", "GeForce RTX 4090"])
def test_exact_gpu(client, search):
    for page in _both(client, search):
        assert "NVIDIA GeForce RTX 4090 (24.0 GB)" in page
        assert "data-box-vram" not in page


def test_search_messages_kept(client):
    for page in _both(client, "RTX"):
        assert "matched multiple entries" in page
        assert "data-box-vram" not in page
    for page in _both(client, "RTX 3060 12GB"):
        assert 'GPU "RTX 3060 12GB" was not found. Using 12 GB inferred from your input.' in page
        assert "data-box-vram" not in page
    for page in _both(client, "Quadro XYZ"):
        assert 'GPU "Quadro XYZ" was not found. Select a listed GPU or enter VRAM manually.' in page
        assert "data-box-vram" not in page


def test_box_number_with_other_hardware_inputs(client):
    page = _get(client, "gpu_id=no_such_gpu&gpu_search=12")
    assert "Manual VRAM" not in page or "was not found" not in page
    assert "matched multiple entries" not in page and "was not found" not in page
    assert "data-box-vram" not in page
    page = _get(client, "vram_gb=abc&gpu_search=12")
    assert "matched multiple entries" not in page and "was not found" not in page
    page = _get(client, "gpu_id=no_such_gpu&gpu_search=0")
    assert "was not found" not in page


def test_vram_gb_wins_over_box_number(client):
    page = _get(client, "vram_gb=24&gpu_search=12")
    assert "Manual VRAM (24 GB)" in page
    assert "data-box-vram" not in page


def test_summary_partial_carries_attribute(client):
    response = client.get("/results/partials/summary?gpu_search=12", headers=HX)
    assert 'data-box-vram="12"' in response.text
    response = client.get("/results/partials/summary?gpu_search=0", headers=HX)
    assert 'data-box-vram="0"' in response.text
    response = client.get("/results/partials/summary?vram_gb=12", headers=HX)
    assert "data-box-vram" not in response.text
