"""T-008: design tokens, the self-hosted font, Google Fonts removed (ADR-0001, ADR-0002)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "src" / "vramsherpa" / "static"
STYLES = STATIC_DIR / "css" / "styles.css"
FONTS_DIR = STATIC_DIR / "fonts"
FONT_NAME = "outfit-v1-latin-wght.woff2"
FONT_URL = f"/static/fonts/{FONT_NAME}"

DARK_MEDIA_SELECTOR = ':root:not([data-theme="light"])'
DARK_SAVED_SELECTOR = ':root[data-theme="dark"]'


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLES.read_text(encoding="utf-8"), flags=re.S)


def _block_body(css: str, open_brace: int) -> str:
    """The text between the brace at `open_brace` and its matching close brace."""
    depth = 0
    for index in range(open_brace, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1 : index]
    raise AssertionError("unbalanced braces in styles.css")


def _declarations(body: str) -> dict[str, str]:
    """Custom properties declared directly in `body`, with whitespace collapsed."""
    found: dict[str, str] = {}
    for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body):
        found[name] = " ".join(value.split())
    return found


def _rule_body(css: str, selector: str, *, after: int = 0) -> str:
    match = re.compile(re.escape(selector) + r"\s*\{").search(css, after)
    assert match, f"no rule for {selector} in styles.css"
    return _block_body(css, match.end() - 1)


def _dark_blocks() -> tuple[dict[str, str], dict[str, str]]:
    css = _css()
    media = re.search(r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{", css)
    assert media, "no prefers-color-scheme: dark block in styles.css"
    media_body = _block_body(css, media.end() - 1)
    return (
        _declarations(_rule_body(media_body, DARK_MEDIA_SELECTOR)),
        _declarations(_rule_body(css, DARK_SAVED_SELECTOR)),
    )


def _font_face_url() -> str:
    match = re.search(r"@font-face\s*\{([^}]*)\}", _css())
    assert match, "no @font-face in styles.css"
    url = re.search(r'url\(\s*["\']?([^"\')]+)["\']?\s*\)', match.group(1))
    assert url, "@font-face has no url()"
    return urljoin("/static/css/styles.css", url.group(1))


def test_stylesheet_starts_with_the_token_section() -> None:
    css = _css().lstrip()
    assert css.startswith("@font-face")
    face = _block_body(css, css.index("{"))
    assert 'font-family: "Outfit"' in face
    assert "font-display: swap" in face
    assert "../fonts/outfit-v1-latin-wght.woff2" in face

    light = _declarations(_rule_body(css, ':root,\n:root[data-theme="light"]'))
    for token in ("--bg", "--ink", "--muted", "--accent", "--btn", "--focus", "--fits-fg"):
        assert token in light
    scales = _declarations(_rule_body(css, ":root"))
    for token in ("--font", "--fs-body", "--sp-4", "--r-pane", "--dur-2", "--tap"):
        assert token in scales


def test_dark_token_blocks_declare_the_same_properties_and_values() -> None:
    media, saved = _dark_blocks()
    assert media, "the prefers-color-scheme dark block declares no properties"
    assert set(media) == set(saved), sorted(set(media) ^ set(saved))
    assert {name: (media[name], saved[name]) for name in media if media[name] != saved[name]} == {}


def test_dark_blocks_cover_every_light_token_that_changes_with_theme() -> None:
    css = _css()
    light = _declarations(_rule_body(css, ':root,\n:root[data-theme="light"]'))
    media, _ = _dark_blocks()
    assert set(light) == set(media), sorted(set(light) ^ set(media))


def test_every_custom_property_used_is_defined() -> None:
    """Pages stay readable between tasks: no var() may point at a name nobody declares."""
    css = _css()
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    used = set(re.findall(r"var\(\s*(--[\w-]+)", css))
    assert sorted(used - defined) == []


def test_asset_version_is_not_the_one_before_the_overhaul() -> None:
    assert main.ASSET_VERSION != "20261024a"
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)


def test_font_is_a_byte_copy_of_the_design_font_and_within_budget() -> None:
    design = ROOT / "docs" / "design" / "fonts" / "outfit-latin-wght-normal.woff2"
    assert (FONTS_DIR / FONT_NAME).read_bytes() == design.read_bytes()
    assert (FONTS_DIR / "LICENSE-outfit.txt").is_file()

    files = [p for p in FONTS_DIR.iterdir() if p.suffix == ".woff2"]
    assert 1 <= len(files) <= 4
    assert sum(p.stat().st_size for p in files) <= 200 * 1024


def test_font_face_url_resolves_to_the_shipped_font() -> None:
    assert _font_face_url() == FONT_URL


@pytest.mark.parametrize("path", ["/", "/how-it-works", "/results?vram_gb=24&context_tokens=4096"])
def test_pages_use_no_third_party_font_host(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert "fonts.googleapis.com" not in response.text
    assert "fonts.gstatic.com" not in response.text
    assert "fonts.googleapis.com" not in STYLES.read_text(encoding="utf-8")


def test_home_preloads_the_font_the_stylesheet_loads(client: TestClient) -> None:
    html = client.get("/").text
    preload = re.search(r"<link\b[^>]*\brel=\"preload\"[^>]*>", html, flags=re.S)
    assert preload, "base.html has no preload link"
    tag = preload.group(0)
    assert re.search(r'\bas="font"', tag)
    assert re.search(r'\btype="font/woff2"', tag)
    assert re.search(r"\bcrossorigin\b", tag)
    href = re.search(r'\bhref="([^"]+)"', tag)
    assert href
    # url_for builds an absolute URL; the path is what has to match the stylesheet's.
    path = urlparse(href.group(1)).path
    assert path == _font_face_url()

    response = client.get(path)
    assert response.status_code == 200
    assert response.content == (FONTS_DIR / FONT_NAME).read_bytes()
