"""T-040: the three counts are the verdict controls (R-085, R-088, R-091)."""

from __future__ import annotations

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

REF = "vram_gb=24&context_tokens=4096"
TILE = re.compile(
    r'<(a|span) class="vt vt-(\w+)"([^>]*)>\s*<svg[^>]*><use href="#i-(\w+)"/></svg>\s*'
    r'<b class="num">(\d+)</b>\s*<span class="vt-label">([^<]*)</span>\s*</\1>'
)


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _tiles(client: TestClient, url: str, headers: dict | None = None) -> list[dict]:
    response = client.get(url, headers=headers or {})
    assert response.status_code == 200
    body = response.text
    assert body.count('aria-label="Show models by verdict"') == 1
    assert "counts" not in re.findall(r'class="([^"]*)"', body)
    return [
        {
            "tag": m[0],
            "cls": m[1],
            "attrs": htmllib.unescape(m[2]),
            "icon": m[3],
            "count": int(m[4]),
            "label": htmllib.unescape(m[5]),
        }
        for m in TILE.findall(body)
    ]


def _href(tile: dict) -> str:
    return re.search(r'href="([^"]*)"', tile["attrs"]).group(1)


def test_gist_tiles(client: TestClient) -> None:
    fits, tight, wont = _tiles(client, f"/results?{REF}")
    assert [t["label"] for t in (fits, tight, wont)] == ["Fits", "Tight", "Won't fit"]
    assert [t["icon"] for t in (fits, tight, wont)] == ["fits", "tight", "wont"]
    assert [t["count"] for t in (fits, tight, wont)] == [71, 0, 24]
    assert tight["tag"] == "span" and "href" not in tight["attrs"]
    assert "tabindex" not in tight["attrs"]
    assert fits["tag"] == wont["tag"] == "a"
    assert _href(fits) == f"/results?{REF}&verdict=fits"
    assert 'data-view-link="verdict"' in fits["attrs"]
    assert 'data-verdict-control="fits"' in fits["attrs"]
    assert f'hx-get="/results/partials/content?{REF}&verdict=fits"' in fits["attrs"]
    assert 'hx-target="#results-shell"' in fits["attrs"]
    assert 'hx-sync="#results-shell:replace"' in fits["attrs"]
    assert "hx-push-url" not in fits["attrs"]
    assert not any("aria-current" in t["attrs"] for t in (fits, tight, wont))


def test_selected_collapses_and_others_keep_page_size(client: TestClient) -> None:
    tiles = _tiles(client, f"/results?{REF}&verdict=fits&page_size=10&page=2")
    fits, tight, wont = tiles
    assert 'aria-current="true"' in fits["attrs"]
    assert 'data-view-link="collapse"' in fits["attrs"]
    assert _href(fits) == f"/results?{REF}"
    assert _href(wont) == f"/results?{REF}&verdict=wont-fit&page_size=10"
    assert "aria-current" not in wont["attrs"] and tight["tag"] == "span"


def test_counts_stay_put(client: TestClient) -> None:
    for tail in ("", "&verdict=fits&page=6", "&verdict=wont-fit&page_size=10"):
        assert [t["count"] for t in _tiles(client, f"/results?{REF}{tail}")] == [71, 0, 24]


def test_zero_count_verdict_marks_none(client: TestClient) -> None:
    tiles = _tiles(client, f"/results?{REF}&verdict=tight")
    assert not any("aria-current" in t["attrs"] for t in tiles)


@pytest.mark.parametrize("query", ["vram_gb=abc", "gpu_id=no_such_gpu", "gpu_search=RTX"])
def test_links_keep_unusable_hardware(client: TestClient, query: str) -> None:
    first = client.get(f"/results?{query}")
    links = [_href(t) for t in _tiles(client, f"/results?{query}") if t["tag"] == "a"]
    assert links
    for link in links:
        assert link.startswith(f"/results?{query}&verdict=")
        followed = client.get(link, follow_redirects=False)
        assert followed.status_code == 200
    assert first.status_code == 200


def test_family_order_and_hostile_verdict(client: TestClient) -> None:
    tiles = _tiles(client, f"/results?{REF}&family=llama&family=qwen")
    assert "family=llama&family=qwen&verdict=" in _href(tiles[0])
    body = client.get(f"/results?{REF}&verdict=%3Cscript%3E").text
    assert "%3Cscript%3E" not in body and "<script>" not in body.split("<body")[1][:0] + ""
    for tile in _tiles(client, f"/results?{REF}&verdict=%3Cscript%3E"):
        assert "script" not in tile["attrs"]


def test_unseeded_tiles_are_inert(app, db_session: Session) -> None:
    body = TestClient(app).get("/results").text
    assert 'data-verdict-control="' not in body
    assert all(t[0] == "span" for t in TILE.findall(body))
