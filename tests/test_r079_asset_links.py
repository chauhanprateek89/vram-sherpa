"""R-079: the overhaul's CSS and JS are requested with a new asset version (rule 21)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

OLD_VERSION = "20261024a"
PAGES = [
    "/",
    "/results?vram_gb=24&context_tokens=4096",
    "/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096",
    "/how-it-works",
]
STATIC_ATTR = re.compile(r'(?:href|src)="([^"]*/static/[^"]*)"')


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _path_query(url: str) -> str:
    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "")


def test_asset_version_is_not_the_pre_overhaul_value() -> None:
    assert main.ASSET_VERSION != OLD_VERSION


@pytest.mark.parametrize("url", PAGES)
def test_css_and_js_links_carry_a_new_version(client: TestClient, url: str) -> None:
    html = client.get(url).text
    links = [_path_query(u) for u in STATIC_ATTR.findall(html)]
    css = [u for u in links if u.startswith("/static/css/styles.css")]
    js = [u for u in links if u.startswith("/static/js/app.js")]
    assert css == [f"/static/css/styles.css?v={main.ASSET_VERSION}"]
    assert js == [f"/static/js/app.js?v={main.ASSET_VERSION}"]
    assert OLD_VERSION not in html


@pytest.mark.parametrize("url", PAGES)
def test_every_static_url_a_page_links_returns_200(client: TestClient, url: str) -> None:
    html = client.get(url).text
    links = {_path_query(u) for u in STATIC_ATTR.findall(html)}
    assert links
    for link in sorted(links):
        assert client.get(link).status_code == 200, link


def test_font_preload_is_the_url_the_stylesheet_loads(client: TestClient) -> None:
    html = client.get("/").text
    preload = re.search(r'<link\s+rel="preload"[^>]*href="([^"]+)"', html, re.S)
    assert preload is not None
    css = client.get(f"/static/css/styles.css?v={main.ASSET_VERSION}").text
    face = re.search(r'@font-face\s*{[^}]*url\("([^"]+)"\)', css, re.S)
    assert face is not None
    # The stylesheet's url() resolves against /static/css/styles.css.
    resolved = urlsplit(
        urlsplit("http://x/static/css/styles.css")._replace(path="/static/css/").geturl()
    )
    resolved_path = re.sub(r"/css/\.\./", "/", f"{resolved.path}{face.group(1)}")
    assert _path_query(preload.group(1)) == resolved_path
