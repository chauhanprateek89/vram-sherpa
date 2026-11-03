"""Chip, Try Q4 and Lower context links keep the verdict and page size (R-031, R-025, R-089)."""

from __future__ import annotations

import html
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _link_tag(text: str, label: str) -> str:
    m = re.search(rf"<a\b[^>]*>\s*{re.escape(label)}\b[^<]*(?:<i[^>]*>[^<]*</i>\s*)?</a>", text)
    assert m, label
    return m.group(0)


def _href(text: str, label: str) -> str:
    return html.unescape(re.search(r'href="([^"]+)"', _link_tag(text, label)).group(1))


def test_family_chip_keeps_verdict_and_size(client: TestClient) -> None:
    url = "/results?vram_gb=24&context_tokens=4096&family=qwen&verdict=fits&page=2&page_size=10"
    text = client.get(url).text
    href = _href(text, "Family: qwen")
    assert "verdict=fits" in href and "page_size=10" in href
    assert "page=" not in href.replace("page_size=", "")
    assert "family=qwen" not in href
    tag = _link_tag(text, "Family: qwen")
    assert "hx-" not in tag


def test_default_size_not_added(client: TestClient) -> None:
    text = client.get("/results?vram_gb=24&context_tokens=4096&family=qwen&verdict=fits").text
    href = _href(text, "Family: qwen")
    assert "verdict=fits" in href and "page_size" not in href


def test_no_verdict_links_unchanged(client: TestClient) -> None:
    text = client.get("/results?vram_gb=24&context_tokens=4096&family=qwen&page=2").text
    assert _href(text, "Family: qwen") == "/results?context_tokens=4096&vram_gb=24"


def test_zero_count_verdict_is_kept(client: TestClient) -> None:
    text = client.get("/results?vram_gb=24&context_tokens=4096&family=qwen&verdict=wont-fit").text
    assert "verdict=wont-fit" in _href(text, "Family: qwen")


def test_empty_state_links_keep_verdict(client: TestClient) -> None:
    # A selected verdict replaces the empty block (R-084), so read the links from the context.
    r = client.get("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8&verdict=tight")
    ctx = r.context
    for key in ("empty_state_q4_url", "empty_state_lower_context_url"):
        href = ctx[key]
        assert "verdict=tight" in href and "page" not in href
    assert "hx-" not in ctx["empty_state_q4_url"]
    gist = client.get("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8")
    for label in ("Try Q4", "Lower context"):
        assert "verdict" not in _href(gist.text, label)
        assert "hx-" not in _link_tag(gist.text, label)
    sized = client.get("/results?vram_gb=1&context_tokens=4096&verdict=wont-fit&page_size=10")
    assert "page_size=10" in sized.context["empty_state_lower_context_url"]


def test_verdict_not_a_filter(client: TestClient) -> None:
    text = client.get(
        "/results?vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4&verdict=fits"
    ).text
    assert "Filters (<span data-fcount>2</span>)" in text


@pytest.mark.parametrize(
    ("ctx", "expected"),
    [
        (4096, "Tight: showing 1-5 of 13 models"),
        (2048, "Tight: showing 1-5 of 9 models"),
        (8192, "No models in Tight with these settings."),
    ],
)
def test_filters_and_context_apply_before_counting(
    client: TestClient, ctx: int, expected: str
) -> None:
    text = client.get(f"/results?vram_gb=8&context_tokens={ctx}&verdict=tight").text
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text))
    assert expected in html.unescape(flat)
