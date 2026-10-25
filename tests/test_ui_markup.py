"""T-010 and T-011: the page shell on every page, and the form tile on `/` and `/results`."""

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


def test_html_has_no_background_so_the_body_colour_cannot_cover_the_backdrop() -> None:
    """A background on html stops body's propagating to the canvas and hides `.backdrop`."""
    css = _css()
    for m in re.finditer(r"([^{}]+)\{([^}]*)\}", css):
        selectors = [s.strip() for s in m.group(1).split(",")]
        if "html" in selectors:
            assert "background" not in m.group(2), m.group(1)
    assert "background: var(--bg)" in _rules(css, "body")[-1]


# ---- T-011: the form tile ----

APP_JS = ROOT / "src" / "vramsherpa" / "static" / "js" / "app.js"


def _form(url: str, client: TestClient) -> _Node:
    form = _one(_tree(client.get(url).text), "form")
    assert form.attrs["id"] == "search-form"
    return form


def _by_id(root: _Node, node_id: str) -> _Node:
    found = [n for n in root.walk() if n.attrs.get("id") == node_id]
    assert len(found) == 1, (node_id, len(found))
    return found[0]


@pytest.mark.parametrize("url", ["/", "/results?vram_gb=24&context_tokens=4096"])
def test_form_tile_uses_the_approved_wording(client: TestClient, url: str) -> None:
    form = _form(url, client)
    assert {"form-pane", "pane"} <= set(form.classes())
    title = _one(form, "h1")
    assert title.all_text() == "Find models that fit your VRAM"
    hint = next(n for n in form.walk() if "hint" in n.classes())
    assert hint.all_text() == "Pick your GPU, or enter your VRAM if it isn't listed."
    text = form.all_text()
    for gone in (
        "I know my GPU",
        "I know my VRAM",
        "Choose either your exact GPU",
        "If your GPU is not listed",
        "leave this blank",
    ):
        assert gone not in text, gone
    assert not [n for n in form.walk() if n.tag == "h2" and "I know" in n.all_text()]
    assert not [n for n in form.walk() if "field-help" in n.classes() and "GPU" in n.all_text()]
    assert _by_id(form, "gpu_search").attrs["placeholder"] == "Start typing a GPU name"


@pytest.mark.parametrize("url", ["/", "/results?vram_gb=24&context_tokens=4096"])
def test_form_tile_has_no_how_it_works_link(client: TestClient, url: str) -> None:
    root = _tree(client.get(url).text)
    form = _one(root, "form")
    assert not [
        n for n in form.walk() if n.tag == "a" and "how-it-works" in n.attrs.get("href", "")
    ]
    page_links = [n for n in root.walk() if n.tag == "a" and n.attrs.get("href") == "/how-it-works"]
    assert len(page_links) == 1, "only the header's link remains"


def test_form_tile_keeps_labels_ids_and_behaviour_hooks(client: TestClient) -> None:
    form = _form("/", client)
    labels = {n.attrs.get("for"): n.all_text() for n in form.walk() if n.tag == "label"}
    assert labels["gpu_search"] == "Search GPU"
    assert labels["vram_gb"] == "Available VRAM (GB)"
    assert _by_id(form, "gpu_id").attrs["type"] == "hidden"
    assert _by_id(form, "gpu_id").attrs["name"] == "gpu_id"
    search = _by_id(form, "gpu_search")
    assert search.attrs["list"] == "gpu-options"
    assert search.attrs["name"] == "gpu_search"
    clear = _by_id(form, "clear-gpu")
    assert clear.tag == "button" and clear.attrs["type"] == "button"
    assert clear.all_text() == "Clear"
    options = _by_id(form, "gpu-options")
    assert options.tag == "datalist"
    assert all(o.attrs["data-gpu-id"] and o.attrs["data-vram-gb"] for o in options.children)
    assert len(options.children) > 10
    legend = next(n for n in form.walk() if n.tag == "legend" and "Context window" in n.all_text())
    assert legend.all_text() == "Context window"
    submit = [n for n in form.walk() if n.tag == "button" and n.attrs.get("type") == "submit"]
    assert [n.all_text() for n in submit if "btn-primary" in n.classes()] == ["Check compatibility"]
    assert not [n for n in form.walk() if "data-chip" in n.attrs]


def test_example_chips_keep_their_labels_and_data_attributes(client: TestClient) -> None:
    form = _form("/", client)
    chips = [n for n in form.walk() if "chip-example" in n.classes()]
    assert [n.all_text() for n in chips] == ["RTX 4060 8GB", "8GB", "24GB"]
    assert all(n.tag == "button" and n.attrs["type"] == "button" for n in chips)
    assert chips[0].attrs.get("data-gpu-id")
    assert float(chips[1].attrs["data-vram-gb"]) == 8
    assert float(chips[2].attrs["data-vram-gb"]) == 24


def test_gpu_search_sits_in_a_combo_wrapper(client: TestClient) -> None:
    form = _form("/", client)
    combos = [n for n in form.walk() if "combo" in n.classes()]
    assert len(combos) == 1
    assert [n.attrs.get("id") for n in combos[0].walk() if n.tag == "input"] == ["gpu_search"]


def test_vram_field_is_a_number_input_with_a_decimal_keypad(client: TestClient) -> None:
    vram = _by_id(_form("/", client), "vram_gb")
    assert vram.attrs["type"] == "number"
    assert vram.attrs["inputmode"] == "decimal"
    assert vram.attrs["min"] == "1"
    assert vram.attrs["step"] == "0.1"
    assert vram.attrs["name"] == "vram_gb"


def test_context_presets_are_a_segmented_control_of_auto_filter_radios(client: TestClient) -> None:
    form = _form("/results?vram_gb=24&context_tokens=4096", client)
    seg = next(n for n in form.walk() if "seg" in n.classes())
    assert "data-thumb" in seg.attrs
    radios = [n for n in seg.walk() if n.tag == "input"]
    assert [r.attrs["value"] for r in radios] == ["2048", "4096", "8192"]
    assert all(r.attrs["type"] == "radio" and r.attrs["name"] == "context_tokens" for r in radios)
    assert all("js-auto-filter" in r.classes() for r in radios)
    assert [r.attrs["value"] for r in radios if "checked" in r.attrs] == ["4096"]
    assert [n.all_text() for n in seg.walk() if n.tag == "span"] == ["2k", "4k", "8k"]
    assert not [n for n in form.walk() if "context-pill" in n.classes()]


@pytest.mark.parametrize("url", ["/", "/results?vram_gb=24&context_tokens=4096"])
def test_the_filters_stay_inside_the_form_for_t012(client: TestClient, url: str) -> None:
    form = _form(url, client)
    names = {n.attrs.get("name") for n in form.walk() if n.tag == "input"}
    assert {"family", "quant_bucket", "min_params_b", "max_params_b", "recommended_only"} <= names


def test_main_is_the_bento_layout_on_the_home_page(client: TestClient) -> None:
    main_node = _one(_tree(client.get("/").text), "main")
    assert main_node.classes() == ["bento"]


def test_input_errors_are_not_rendered_inside_the_form(client: TestClient) -> None:
    html = client.get("/results?vram_gb=abc").text
    assert html.count('role="alert"') == 1
    assert html.count("Invalid value for vram_gb.") == 1
    assert html.count("Please fix the following input issues:") == 1
    form = _one(_tree(html), "form")
    assert "Invalid value" not in form.all_text()
    assert not [n for n in form.walk() if n.attrs.get("role") in {"alert", "status"}]


def test_gpu_search_feedback_is_not_rendered_inside_the_form(client: TestClient) -> None:
    html = client.get("/results?gpu_search=RTX").text
    assert html.count("matched multiple entries") == 1
    root = _tree(html)
    status = [n for n in root.walk() if n.attrs.get("role") == "status"]
    assert len(status) == 1
    assert "matched multiple entries" in status[0].all_text()
    form = _one(root, "form")
    assert "matched multiple entries" not in form.all_text()
    assert not [n for n in form.walk() if "field-warning" in n.classes()]


def test_context_segmented_control_is_filled_without_javascript() -> None:
    css = _css()
    checked = [
        body for body in _rules(css, ".seg label:has(input:checked) span") if "background" in body
    ]
    assert checked and "var(--pressed)" in checked[0]
    assert "var(--on-pressed)" in checked[0]
    # With the thumb present the label stops painting, so the two do not double up.
    assert "background: transparent" in _rules(css, ".has-thumb label:has(input:checked) span")[0]


def test_segmented_control_targets_and_field_text_meet_the_phone_sizes() -> None:
    css = _css()
    assert "min-height: 46px" in _rules(css, ".seg span")[0]
    field = _rules(css, ".form-pane input[type=text], .form-pane input[type=number]")[0]
    assert "font-size: 16px" in field
    assert "min-height: 52px" in field
    assert "min-height: var(--tap)" in _rules(css, ".btn-clear, .chips .chip")[0]
    assert "min-height: 54px" in _rules(css, ".btn-primary")[0]


def test_thumb_has_no_transition_under_reduced_motion() -> None:
    css = _css()
    reduced = re.findall(
        r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, flags=re.S
    )
    assert any(".thumb" in block and "transition: none" in block for block in reduced)


def test_old_hero_form_rules_are_gone() -> None:
    css = _css()
    for dead in (
        ".hero-panel",
        ".hero-paths",
        ".input-path",
        ".gpu-input-row",
        ".hero-copy",
        ".hero-actions",
        ".context-pill",
        ".example-chips",
        ".search-form",
    ):
        assert dead not in css, dead


def test_app_js_places_a_thumb_and_replaces_it_on_every_layout_change() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert "[data-thumb]" in js
    assert '"thumb"' in js and "has-thumb" in js
    for trigger in ('"change"', '"resize"', "fonts.ready", '"htmx:afterSettle"'):
        assert trigger in js, trigger
    # The placement is progressive: it reads layout and writes custom properties only.
    for prop in ("--x", "--y", "--w", "--h"):
        assert prop in js, prop


def test_asset_version_is_a_date_and_letter_and_reaches_the_page(client: TestClient) -> None:
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)
    assert main.ASSET_VERSION > "20261025d"
    assert f"styles.css?v={main.ASSET_VERSION}" in client.get("/").text
