"""T-010: the page shell on every page: backdrop, glass header and footer (R-068, R-076, R-077)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

ROOT = Path(__file__).resolve().parents[1]
STYLES = ROOT / "src" / "vramsherpa" / "static" / "css" / "styles.css"
TEMPLATES = ROOT / "src" / "vramsherpa" / "templates"

PAGES = {
    "/": "Estimator",
    "/results?vram_gb=24&context_tokens=4096": "Estimator",
    "/models/model_qwen25_7b_instruct": "Estimator",
    "/how-it-works": "How it works",
}


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


class _Node:
    def __init__(self, tag: str, attrs: dict[str, str | None], parent: _Node | None) -> None:
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: list[_Node] = []
        self.text = ""

    def classes(self) -> list[str]:
        return (self.attrs.get("class") or "").split()

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def all_text(self) -> str:
        return " ".join("".join(n.text for n in self.walk()).split())


class _Tree(HTMLParser):
    VOID = {"meta", "link", "br", "img", "input", "hr"}

    def __init__(self) -> None:
        super().__init__()
        self.root = _Node("root", {}, None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs) -> None:
        node = _Node(tag, dict(attrs), self.cur)
        self.cur.children.append(node)
        if tag not in self.VOID:
            self.cur = node

    def handle_endtag(self, tag) -> None:
        node = self.cur
        while node is not self.root and node.tag != tag:
            node = node.parent  # type: ignore[assignment]
        if node is not self.root:
            self.cur = node.parent  # type: ignore[assignment]

    def handle_data(self, data) -> None:
        self.cur.text += data


def _tree(html: str) -> _Node:
    parser = _Tree()
    parser.feed(html)
    return parser.root


def _one(root: _Node, tag: str) -> _Node:
    found = [n for n in root.walk() if n.tag == tag]
    assert len(found) == 1, (tag, len(found))
    return found[0]


def _shell(html: str) -> tuple[str, str]:
    """The header and footer markup, with `aria-current` and the active class removed."""
    parts = []
    for tag in ("header", "footer"):
        match = re.search(rf"<{tag}\b.*?</{tag}>", html, re.S)
        assert match, tag
        text = re.sub(r'\s*aria-current="page"', "", match.group(0))
        text = re.sub(r"\bis-active\b", "", text)
        parts.append(" ".join(text.split()))
    return parts[0], parts[1]


@pytest.mark.parametrize("url", PAGES)
def test_page_wraps_header_main_footer_in_backdrop_and_page(client: TestClient, url: str) -> None:
    root = _tree(client.get(url).text)
    body = _one(root, "body")
    kids = [n for n in body.children if n.tag in {"div", "script"}]
    backdrop = kids[0]
    assert backdrop.classes() == ["backdrop"]
    assert backdrop.attrs.get("aria-hidden") == "true"
    page = kids[1]
    assert page.classes() == ["page"]
    assert [n.tag for n in page.children] == ["header", "main", "footer"]
    header, _, footer = page.children
    assert header.classes() == ["top", "pane"]
    assert footer.classes() == ["footer", "pane", "lite"]


@pytest.mark.parametrize("url", PAGES)
def test_header_has_brand_nav_and_theme_toggle(client: TestClient, url: str) -> None:
    header = _one(_tree(client.get(url).text), "header")
    brand = next(n for n in header.walk() if "brand" in n.classes())
    assert brand.attrs["href"] == "/"
    assert brand.all_text() == "VRAM Sherpa"
    assert any(n.tag == "svg" for n in brand.walk())

    nav = _one(header, "nav")
    links = [n for n in nav.walk() if n.tag == "a"]
    assert [(n.all_text(), n.attrs["href"]) for n in links] == [
        ("Estimator", "/"),
        ("How it works", "/how-it-works"),
    ]

    toggle = next(n for n in header.walk() if n.attrs.get("id") == "theme-toggle")
    assert toggle.tag == "button"
    assert toggle.attrs["type"] == "button"
    assert toggle.attrs["aria-pressed"] in {"true", "false"}
    assert toggle.all_text() in {"Dark mode", "Light mode"}
    assert "theme" in toggle.classes()


@pytest.mark.parametrize(("url", "current"), list(PAGES.items()))
def test_aria_current_marks_one_link(client: TestClient, url: str, current: str) -> None:
    nav = _one(_tree(client.get(url).text), "nav")
    marked = [n.all_text() for n in nav.walk() if n.attrs.get("aria-current") == "page"]
    assert marked == [current]
    active = [n.all_text() for n in nav.walk() if "is-active" in n.classes()]
    assert active == [current]


def test_header_and_footer_are_identical_on_every_page(client: TestClient) -> None:
    shells = {url: _shell(client.get(url).text) for url in PAGES}
    assert len(set(shells.values())) == 1, "header or footer differs between pages"


@pytest.mark.parametrize("url", PAGES)
def test_footer_shows_the_three_values(client: TestClient, url: str) -> None:
    footer = _one(_tree(client.get(url).text), "footer")
    spans = [n.all_text() for n in footer.children if n.tag == "span"]
    assert spans == [
        "Catalog version: 2026-10-23",
        f"Estimation policy: {main.ESTIMATION_POLICY_VERSION}",
        "Last updated: 2026-10-23",
    ]


def test_main_class_comes_from_a_template_block() -> None:
    env = main.templates.env
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert '<main class="{% block main_class %}{% endblock %}">' in base
    source = '{% extends "base.html" %}{% block main_class %}bento{% endblock %}'
    html = env.from_string(source).render(
        current_path="/",
        catalog_version="v",
        estimation_policy_version="p",
        catalog_last_updated="v",
        asset_version="x",
        request=SimpleNamespace(url_for=lambda name, **kw: "/static/x"),
    )
    assert '<main class="bento">' in html


# ---- styles ----


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLES.read_text(encoding="utf-8"), flags=re.S)


def _rules(css: str, selector: str) -> list[str]:
    return [m.group(1) for m in re.finditer(re.escape(selector) + r"\s*\{([^}]*)\}", css)]


def test_glass_has_both_solid_fallbacks() -> None:
    css = _css()
    supports = re.search(
        r"@supports not \(\(backdrop-filter: blur\(1px\)\)[^{]*\{([^@]*)\}\s*\n", css
    )
    assert supports, "no @supports not (backdrop-filter) fallback"
    assert "--panel-solid" in supports.group(1)
    reduced = re.search(r"@media \(prefers-reduced-transparency: reduce\)\s*\{(.*)\}\s*\n", css)
    assert reduced, "no prefers-reduced-transparency fallback"
    assert "var(--panel-solid)" in reduced.group(1)
    assert "backdrop-filter: none" in reduced.group(1)


def test_lite_pane_is_blur_free_on_a_phone() -> None:
    """Glass budget: phone width blurs the header only; the footer (`lite`) blurs from 900 px."""
    css = _css()
    base = _rules(css, ".lite")
    assert base and "backdrop-filter: none" in base[0]
    wide = re.search(r"@media \(min-width: 900px\)\s*\{\s*\.lite\s*\{([^}]*)\}", css)
    assert wide and "blur(" in wide.group(1)


def test_backdrop_is_fixed_and_not_animated() -> None:
    body = _rules(_css(), ".backdrop")[0]
    assert "position: fixed" in body
    assert "animation" not in body
    assert "filter" not in body


def test_header_targets_are_44px_and_the_toggle_icon_is_a_mask_on_aria_pressed() -> None:
    css = _css()
    assert "--tap: 44px" in css
    for selector in (".brand", ".top nav a", ".theme"):
        assert "min-height: var(--tap)" in _rules(css, selector)[0], selector
    assert "mask: var(--ic-moon)" in _rules(css, ".theme::before")[0]
    pressed = _rules(css, '.theme[aria-pressed="true"]::before')[0]
    assert "mask-image: var(--ic-sun)" in pressed


def test_old_header_and_results_layout_rules_are_gone() -> None:
    css = _css()
    for dead in (".topbar", ".container", ".theme-toggle", ".nav-link", ".results-layout"):
        assert dead not in css, dead


def test_asset_version_changed_for_the_shell() -> None:
    assert main.ASSET_VERSION not in {"20261024a", "20261025a", "20261025b"}
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)
