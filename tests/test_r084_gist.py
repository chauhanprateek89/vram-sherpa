"""The gist: /results opens with no card, the headline, the notes and the zero message (R-084)."""

from __future__ import annotations

import html as htmllib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

REF = "vram_gb=24&context_tokens=4096"
HX = {"HX-Request": "true"}
GIST_PATHS = ["/results", "/results/partials/content", "/results/partials/list"]
CHOOSE = "Choose Fits, Tight or Won't fit to see the models."
PICK = "Pick a quantisation on any card to see how it changes."


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _get(client: TestClient, url: str, headers: dict | None = None) -> str:
    response = client.get(url, headers=headers or {})
    assert response.status_code == 200, url
    return htmllib.unescape(response.text)


@pytest.mark.parametrize("path", GIST_PATHS)
@pytest.mark.parametrize("headers", [{}, HX])
def test_gist_has_no_card(client: TestClient, path: str, headers: dict) -> None:
    if path != "/results" and not headers:
        pytest.skip("partials redirect a page load")
    body = _get(client, f"{path}?{REF}", headers)
    assert "data-model-card" not in body
    assert "How this is estimated" not in body
    if path != "/results/partials/list":
        assert 'id="summary-banner"' in body
        assert "Manual VRAM (24 GB)" in body
    else:
        assert body.strip() == ""


@pytest.mark.parametrize(
    "query",
    [REF, f"{REF}&verdict=fits&page=6", f"{REF}&verdict=wont-fit&page=2"],
)
def test_headline_is_fits_plus_tight_in_every_view(client: TestClient, query: str) -> None:
    assert "71 models fit or are tight" in _get(client, f"/results?{query}")


def test_headline_singular(client: TestClient) -> None:
    assert "1 model fit or are tight" in _get(client, "/results?vram_gb=3&family=phi")


def test_notes(client: TestClient) -> None:
    gist = _get(client, f"/results?{REF}")
    assert CHOOSE in gist and PICK not in gist
    picked = _get(client, f"/results?{REF}&verdict=fits")
    assert PICK in picked and CHOOSE not in picked
    for body in (gist, picked, _get(client, "/results?vram_gb=1")):
        assert "Only Fits and Tight models are listed." not in body
    assert CHOOSE in _get(client, "/results?vram_gb=1")


def test_zero_count_verdict_message(client: TestClient) -> None:
    body = _get(client, f"/results?{REF}&verdict=tight")
    assert 'data-verdict="tight"' in body
    msg = "No models in Tight with these settings."
    assert msg in body
    assert 'class="notice" role="status"' in body
    assert body.index(msg) < body.index('class="verdicts"')
    assert "data-model-card" not in body
    wont = _get(client, "/results?vram_gb=1&verdict=fits")
    assert "No models in Fits with these settings." in wont
    assert "No models in" not in _get(client, f"/results?{REF}&verdict=fits")


@pytest.mark.parametrize(
    "url", ["/results?vram_gb=1", "/results?vram_gb=24&min_params_b=5&max_params_b=1"]
)
def test_empty_state_on_the_gist(client: TestClient, url: str) -> None:
    body = _get(client, url)
    assert "No models currently fit this setup." in body
    assert 'class="sr-only">Results<' in body
    if "vram_gb=1" in url:
        assert "models in Won't fit, hidden by default." in body


def test_empty_state_absent_with_models_and_with_wont_fit_selected(client: TestClient) -> None:
    gist = _get(client, f"/results?{REF}")
    assert "No models currently fit this setup." not in gist
    assert "sr-only" not in gist.split('id="results-list-shell"')[1]
    listing = _get(client, "/results?vram_gb=1&verdict=wont-fit")
    assert "No models currently fit this setup." not in listing
    assert listing.count("data-model-card") == 5
    assert 'class="sr-only">Results<' in listing


def test_unseeded_catalog_shows_r026_and_the_message(app, db_session: Session) -> None:
    client = TestClient(app)
    body = _get(client, f"/results?{REF}")
    assert "Catalog is not initialized yet." in body
    assert "Seed the catalog data, then refresh this page." in body
    assert 'class="note"' not in body
    named = _get(client, f"/results?{REF}&verdict=fits")
    assert "Catalog is not initialized yet." in named
    assert "No models in Fits with these settings." in named


def test_no_verdict_page_and_page_size_change_nothing(client: TestClient) -> None:
    assert _get(client, f"/results?{REF}&page=3&page_size=10") == _get(client, f"/results?{REF}")


def test_partial_content_page_load_ends_on_the_gist(client: TestClient) -> None:
    response = client.get(f"/results/partials/content?{REF}", follow_redirects=True)
    assert response.url.path == "/results"
    assert 'id="summary-banner"' in response.text
    assert "data-model-card" not in response.text


@pytest.mark.parametrize(
    "query",
    [
        REF,
        "vram_gb=8&context_tokens=2048",
        f"{REF}&verdict=fits",
        f"{REF}&verdict=fits&page=6",
        f"{REF}&verdict=fits&page=15",
        f"{REF}&verdict=fits&page_size=10",
        f"{REF}&verdict=wont-fit",
        f"{REF}&verdict=tight",
        "vram_gb=8&context_tokens=2048&verdict=tight",
    ],
)
def test_partial_content_body_is_inside_the_page(client: TestClient, query: str) -> None:
    fragment = client.get(f"/results/partials/content?{query}", headers=HX).text
    page = client.get(f"/results?{query}").text
    assert fragment in page
