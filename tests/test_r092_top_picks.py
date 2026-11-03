"""Top Picks open on the gist and show the card's verdict and three figures (R-092)."""

from __future__ import annotations

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

REF = "vram_gb=24&context_tokens=4096"


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _get(client: TestClient, url: str) -> str:
    response = client.get(url)
    assert response.status_code == 200, url
    return response.text


def _tiles(body: str) -> list[str]:
    return re.findall(r'<article class="pick".*?</article>', body, re.S)


def _text(fragment: str) -> list[str]:
    stripped = re.sub(r"<svg.*?</svg>", "", fragment, flags=re.S)
    parts = re.split(r"<[^>]+>", stripped)
    return [htmllib.unescape(p).strip() for p in parts if p.strip()]


def test_each_pick_shows_verdict_and_three_figures(client: TestClient) -> None:
    tiles = _tiles(_get(client, f"/results?{REF}"))
    assert len(tiles) == 3
    fig = re.search(r'<div class="pick-fig">.*?</div>\s*</div>', tiles[0], re.S)
    assert fig and tiles[0].index("pick-desc") < tiles[0].index("pick-fig")
    assert _text(tiles[0])[:3] == [
        "Best quality that fits",
        "DeepSeek-Coder-V2-Lite-Instruct",
        "15.7B · Q4",
    ]
    assert _text(fig.group(0)) == ["Fits", "8.6 GB", "spare", "15.36 GB needed", "24.0 GB"]
    for tile in tiles:
        assert "GB over" not in tile and 'class="track"' not in tile and "why-panel" not in tile


def test_pick_figures_match_the_cards_pane(client: TestClient) -> None:
    tile = _tiles(_get(client, f"/results?{REF}"))[0]
    name = _text(tile)[1]
    found = None
    for page in range(1, 10):
        body = _get(client, f"/results?{REF}&verdict=fits&page={page}")
        cards = re.findall(r"<article[^>]*data-model-card.*?</article>", body, re.S)
        found = next((c for c in cards if name in c), None)
        if found or not cards:
            break
    assert found, "the pick's model is on a Fits page"
    panes = re.findall(r'<div\s+class="fit-pane".*?(?=<details class="why-panel")', found, re.S)
    texts = [_text(p) for p in panes]
    assert ["Fits", "8.6 GB", "spare", "15.36 GB needed", "24.0 GB"] in texts


def test_open_on_gist_closed_with_verdict_and_same_picks(client: TestClient) -> None:
    gist = _get(client, f"/results?{REF}")
    tag = re.search(r'<details class="picks[^>]*>', gist).group(0)
    assert " open" in tag and "data-open-wide" not in tag
    listed = _get(client, f"/results?{REF}&verdict=fits")
    tag = re.search(r'<details class="picks[^>]*>', listed).group(0)
    assert " open>" not in tag and "data-open-wide" in tag
    assert _tiles(listed) == _tiles(gist)
    paged = _get(client, f"/results?{REF}&verdict=fits&page=2")
    assert _tiles(paged) == _tiles(gist)


def test_picks_hidden_where_r024_says(client: TestClient) -> None:
    assert 'class="picks' not in _get(client, "/results?vram_gb=2&context_tokens=4096")
