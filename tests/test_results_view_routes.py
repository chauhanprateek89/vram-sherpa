"""Routes read the results view: verdict, page and page size (R-086, R-087, R-088, R-090)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

REF = "vram_gb=24&context_tokens=4096"
SMALL = "vram_gb=8&context_tokens=2048"
HX = {"HX-Request": "true"}


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _cards(client: TestClient, url: str, headers: dict | None = None) -> int:
    response = client.get(url, headers=headers or {})
    assert response.status_code == 200
    return response.text.count("data-model-card")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (f"/results?{REF}&verdict=fits", 5),
        (f"/results?{REF}&verdict=fits&page=15", 1),
        (f"/results?{REF}&verdict=fits&page_size=10", 10),
        (f"/results?{REF}&verdict=fits&page_size=10&page=8", 1),
        (f"/results?{SMALL}&verdict=tight", 5),
        (f"/results?{SMALL}&verdict=tight&page=2", 4),
        (f"/results?{SMALL}&verdict=tight&page_size=10", 9),
        (f"/results?{SMALL}&family=llama&verdict=tight", 4),
        (f"/results/partials/list?{REF}&verdict=fits", 5),
        (f"/results/partials/summary?{REF}&verdict=fits", 0),
    ],
)
def test_card_counts(client: TestClient, url: str, expected: int) -> None:
    assert _cards(client, url) == expected


def test_no_verdict_passes_no_cards(client: TestClient) -> None:
    assert _cards(client, f"/results?{REF}") == 0


@pytest.mark.parametrize("page", ["abc", "0", "-1", "1.5"])
def test_bad_page_shows_page_one(client: TestClient, page: str) -> None:
    base = client.get(f"/results?{REF}&verdict=fits").text
    r = client.get(f"/results?{REF}&verdict=fits&page={page}")
    assert r.status_code == 200
    assert r.text.count("data-model-card") == 5
    names = lambda t: re.findall(r"(?s)data-model-card.*?<h3>\s*<a [^>]*>([^<]+)", t)  # noqa: E731
    assert names(r.text) == names(base)


def test_huge_page_clamps_without_redirect(client: TestClient) -> None:
    last = client.get(f"/results?{REF}&verdict=fits&page=15", follow_redirects=False)
    for page in ("99", "9" * 5000):
        r = client.get(f"/results?{REF}&verdict=fits&page={page}", follow_redirects=False)
        assert r.status_code == 200
        assert "location" not in r.headers
        assert r.text.count("data-model-card") == 1
        names = re.findall(r"(?s)data-model-card.*?<h3>\s*<a [^>]*>([^<]+)", r.text)
        assert names == re.findall(r"(?s)data-model-card.*?<h3>\s*<a [^>]*>([^<]+)", last.text)


@pytest.mark.parametrize("size", ["7", "abc", "1000", ""])
def test_bad_page_size_is_five(client: TestClient, size: str) -> None:
    r = client.get(f"/results?{REF}&verdict=fits&page_size={size}")
    assert r.status_code == 200
    assert r.text.count("data-model-card") == 5
    assert "data-page-size" not in r.text


def test_last_verdict_wins(client: TestClient) -> None:
    r = client.get(f"/results?{REF}&verdict=tight&verdict=fits")
    assert 'data-verdict="fits"' in r.text


@pytest.mark.parametrize("verdict", ["green", "FITS", ""])
def test_bad_verdict_selects_nothing(client: TestClient, verdict: str) -> None:
    r = client.get(f"/results?{REF}&verdict={verdict}")
    assert r.status_code == 200
    assert "data-verdict" not in r.text


def test_script_verdict_is_not_echoed(client: TestClient) -> None:
    r = client.get("/results?vram_gb=24&verdict=%3Cscript%3E")
    assert r.status_code == 200
    assert "data-verdict" not in r.text
    assert "alert" not in r.text
    assert r.text.count("<script>") == client.get("/results?vram_gb=24").text.count("<script>")


def test_banner_carries_canonical_view(client: TestClient) -> None:
    r = client.get(f"/results?{REF}&verdict=fits&page_size=10&page=2")
    assert 'data-verdict="fits"' in r.text
    assert 'data-page-size="10"' in r.text
    zero = client.get("/results?vram_gb=1&context_tokens=2048&verdict=fits")
    assert 'data-verdict="fits"' in zero.text
    assert "data-model-card" not in zero.text


def test_wont_fit_cards(client: TestClient) -> None:
    first = client.get(f"/results?{REF}&verdict=wont-fit").text
    assert first.count("data-model-card") == 5
    assert re.findall(r"(?s)data-model-card.*?<h3>\s*<a [^>]*>([^<]+)", first)[0] == "Qwen3-30B-A3B"
    for needle in (
        "bar-wont",
        "width: 100.0%",
        " GB needed",
        "data-oversize",
        "How this is estimated",
    ):
        assert needle in first
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", first))
    assert re.search(r"4\.4 GB over", flat)
    last = client.get(f"/results?{REF}&verdict=wont-fit&page=5").text
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", last))
    assert (
        re.findall(r"(?s)data-model-card.*?<h3>\s*<a [^>]*>([^<]+)", last)[-1]
        == "DeepSeek-V4-Pro-0813"
    )
    assert "1486.4 GB over" in flat


def test_oversize_pressed_rule_follows_base_rule() -> None:
    text = (Path(main.__file__).parent / "static" / "css" / "styles.css").read_text()
    base = text.index(".quants button[data-oversize] {")
    pressed = text.index('.quants button[data-oversize][aria-pressed="true"]')
    assert pressed > base
    block = text[pressed : text.index("}", pressed)]
    assert "color: var(--on-pressed)" in block


def test_partial_content_redirect_and_push_url(client: TestClient) -> None:
    query = "vram_gb=24&context_tokens=4096&family=qwen&family=llama&verdict=fits"
    r = client.get(f"/results/partials/content?{query}", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"/results?{query}"
    r = client.get(f"/results/partials/content?{query}", headers=HX)
    assert r.status_code == 200
    assert r.headers["hx-push-url"] == f"/results?{query}"
    r = client.get("/results?verdict=fits", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/"
