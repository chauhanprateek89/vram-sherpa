"""T-009: the logo mark in the header and the favicons under /static/img (R-068, R-076)."""

from __future__ import annotations

import re
import struct
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "src" / "vramsherpa" / "static" / "img"
TEMPLATES = ROOT / "src" / "vramsherpa" / "templates"
MARKS = ROOT / "docs" / "design" / "marks"

QUERY = "vram_gb=24&context_tokens=4096"
FULL_PAGES = [
    "/",
    f"/results?{QUERY}",
    "/models/model_qwen25_7b_instruct",
    "/how-it-works",
]

# R-042: the route paths, nothing added.
R042_PATHS = {
    "/",
    "/results",
    "/results/partials/content",
    "/results/partials/summary",
    "/results/partials/list",
    "/results/why/{variant_id}",
    "/models/{model_id}",
    "/how-it-works",
    "/healthz",
    "/static",
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


class _Head(HTMLParser):
    """Collects the `<link>` tags and the header brand link of a page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str | None]] = []
        self.brand_depth = 0
        self.brand_attrs: dict[str, str | None] | None = None
        self.svg_in_brand: dict[str, str | None] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link":
            self.links.append(values)
        if tag == "a" and "brand" in (values.get("class") or "").split():
            self.brand_attrs = values
            self.brand_depth = 1
        elif self.brand_depth:
            if tag == "svg":
                self.svg_in_brand = values
            self.brand_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.brand_depth:
            self.brand_depth -= 1


def _parse(html: str) -> _Head:
    parser = _Head()
    parser.feed(html)
    return parser


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == PNG_SIGNATURE
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])  # type: ignore[return-value]


def _link(page: _Head, rel: str, **match: str) -> dict[str, str | None]:
    found = [
        link
        for link in page.links
        if link.get("rel") == rel and all(link.get(k) == v for k, v in match.items())
    ]
    assert len(found) == 1, (rel, match, page.links)
    return found[0]


def _path_and_query(link: dict[str, str | None]) -> str:
    """`url_for` yields an absolute URL; the host is not under test."""
    parts = urlparse(link["href"] or "")
    return f"{parts.path}?{parts.query}"


@pytest.mark.parametrize("url", FULL_PAGES)
def test_header_brand_link_holds_the_inline_mark(client: TestClient, url: str) -> None:
    response = client.get(url)
    assert response.status_code == 200
    page = _parse(response.text)

    assert page.brand_attrs is not None
    assert page.brand_attrs["href"] == "/"
    assert page.svg_in_brand is not None
    assert page.svg_in_brand["aria-hidden"] == "true"
    assert page.svg_in_brand["focusable"] == "false"
    assert page.svg_in_brand["viewbox"] == "0 0 34 34"
    # The wordmark stays in the same link, as text.
    assert re.search(r"</svg>\s*<span[^>]*>VRAM Sherpa</span>\s*</a>", response.text)
    # One mark per page: the gradient ids in it must not repeat.
    assert response.text.count('id="vsA"') == 1


def test_template_is_the_approved_mark_with_a11y_attributes() -> None:
    approved = (MARKS / "final-mark.svg").read_text(encoding="utf-8").strip()
    template = (TEMPLATES / "logo_mark.html").read_text(encoding="utf-8")
    template = re.sub(r"\{#.*?#\}", "", template, flags=re.S).strip()

    def inner(svg: str) -> str:
        return re.sub(r"^<svg[^>]*>|</svg>$", "", svg)

    assert inner(template) == inner(approved)
    opening = re.match(r"<svg[^>]*>", template)
    assert opening is not None
    assert 'aria-hidden="true"' in opening.group(0)
    assert 'focusable="false"' in opening.group(0)
    # Colour comes from the mark's own gradients; no hard-coded page colour is added to the tag.
    assert "style=" not in opening.group(0)


@pytest.mark.parametrize("url", FULL_PAGES)
def test_head_links_the_three_icons_with_the_asset_version(client: TestClient, url: str) -> None:
    page = _parse(client.get(url).text)
    version = f"?v={main.ASSET_VERSION}"

    svg = _link(page, "icon", type="image/svg+xml")
    png = _link(page, "icon", type="image/png")
    touch = _link(page, "apple-touch-icon")

    assert _path_and_query(svg) == f"/static/img/favicon.svg{version}"
    assert _path_and_query(png) == f"/static/img/favicon-32.png{version}"
    assert png["sizes"] == "32x32"
    assert _path_and_query(touch) == f"/static/img/apple-touch-icon.png{version}"
    assert touch["sizes"] == "180x180"


def test_every_icon_url_returns_200(client: TestClient) -> None:
    page = _parse(client.get("/").text)
    hrefs = [link["href"] for link in page.links if link.get("rel") in {"icon", "apple-touch-icon"}]
    assert len(hrefs) == 3
    for href in hrefs:
        assert href is not None
        response = client.get(urlparse(href).path)
        assert response.status_code == 200, href
        assert len(response.content) > 0


def test_favicon_svg_is_a_copy_of_the_approved_one() -> None:
    assert (IMG_DIR / "favicon.svg").read_bytes() == (MARKS / "final-favicon.svg").read_bytes()


def test_png_icons_have_the_stated_sizes() -> None:
    assert _png_size((IMG_DIR / "favicon-32.png").read_bytes()) == (32, 32)
    assert _png_size((IMG_DIR / "apple-touch-icon.png").read_bytes()) == (180, 180)


def test_no_route_was_added_and_favicon_ico_is_404(app: FastAPI, client: TestClient) -> None:
    # FastAPI's own schema and docs routes are the framework's, not the app's; they predate R-068.
    framework = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    paths = {getattr(route, "path", None) for route in app.routes} - framework
    assert paths == R042_PATHS
    assert client.get("/favicon.ico").status_code == 404


def test_asset_version_is_new_for_this_task() -> None:
    # Later tasks bump it again, so only the earlier values are ruled out (as in test_styles.py).
    assert main.ASSET_VERSION not in {"20261024a", "20261025a", "20261025b"}
