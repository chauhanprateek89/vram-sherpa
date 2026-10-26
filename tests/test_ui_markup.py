"""T-010 to T-013: the page shell, the form tile, its filters and the answer tile."""

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


FILTERED = "/results?vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4"
APP_JS = ROOT / "src" / "vramsherpa" / "static" / "js" / "app.js"


def _filters(url: str, client: TestClient) -> tuple[_Node, _Node]:
    form = _form(url, client)
    found = [n for n in form.walk() if n.tag == "details" and "filters" in n.classes()]
    assert len(found) == 1
    return form, found[0]


@pytest.mark.parametrize("url", ["/", "/results?vram_gb=24&context_tokens=4096", FILTERED])
def test_filters_are_a_closed_disclosure_at_the_foot_of_the_form(
    client: TestClient, url: str
) -> None:
    form, filters = _filters(url, client)
    assert "open" not in filters.attrs, "closed at every width (ADR-0006 point 4, Gate 2)"
    assert filters.parent is form
    assert form.children[-1] is filters, "the foot of the form tile"
    assert [c.tag for c in filters.children if c.tag == "summary"] == ["summary"]
    assert next(c for c in form.children if c.tag == "button").all_text() == "Check compatibility"
    names = {n.attrs.get("name") for n in filters.walk() if n.tag == "input"}
    assert names == {"family", "quant_bucket", "min_params_b", "max_params_b", "recommended_only"}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/", 0),
        ("/results?vram_gb=24&context_tokens=4096", 0),
        (FILTERED, 2),
        (FILTERED + "&min_params_b=3&max_params_b=14&recommended_only=true", 5),
        ("/results?vram_gb=24&family=qwen&family=llama&min_params_b=abc", 2),
    ],
)
def test_summary_counts_the_active_filter_chips(
    client: TestClient, url: str, expected: int
) -> None:
    html = client.get(url).text
    assert f"<span>Filters (<span data-fcount>{expected}</span>)</span>" in html
    root = _tree(html)
    chips = [n for n in root.walk() if "chip-link" in n.classes()]
    assert len(chips) == expected


def test_filters_summary_has_the_funnel_icon_and_the_count(client: TestClient) -> None:
    _, filters = _filters(FILTERED, client)
    summary = next(c for c in filters.children if c.tag == "summary")
    assert any(n.tag == "svg" and n.attrs.get("aria-hidden") == "true" for n in summary.walk())
    # The nested count span makes the tree helper put the count after the closing bracket.
    assert summary.all_text().replace(" ", "") == "Filters()2"


@pytest.mark.parametrize("url", ["/", FILTERED])
def test_filters_keep_every_control_label_and_title(client: TestClient, url: str) -> None:
    form, filters = _filters(url, client)
    legends = {n.all_text(): n for n in filters.walk() if n.tag == "legend"}
    assert set(legends) == {"Family", "Quantization"}
    assert legends["Quantization"].attrs["title"] == (
        "Quantization level. Lower precision (like Q4) usually uses less VRAM."
    )
    labels = {
        n.attrs.get("for"): n for n in filters.walk() if n.tag == "label" and n.attrs.get("for")
    }
    assert labels["min_params_b"].all_text() == "Min params (B)"
    assert labels["min_params_b"].attrs["title"] == (
        "Show only models with parameter count greater than or equal to this value."
    )
    assert labels["max_params_b"].all_text() == "Max params (B)"
    assert labels["max_params_b"].attrs["title"] == (
        "Show only models with parameter count less than or equal to this value."
    )
    checks = [n for n in filters.walk() if n.tag == "input" and n.attrs.get("type") == "checkbox"]
    solo = next(c for c in checks if c.attrs["name"] == "recommended_only")
    assert solo.parent.all_text() == "Recommended only"
    assert "choice" in solo.parent.classes()
    fams = [c.attrs["value"] for c in checks if c.attrs["name"] == "family"]
    quants = [c.attrs["value"] for c in checks if c.attrs["name"] == "quant_bucket"]
    assert fams == sorted(fams) and "qwen" in fams
    assert quants == ["FP16", "Q4", "Q5", "Q8"]


def test_checked_filters_are_checked_in_the_markup(client: TestClient) -> None:
    _, filters = _filters(FILTERED, client)
    checked = [
        (n.attrs["name"], n.attrs["value"])
        for n in filters.walk()
        if n.tag == "input" and "checked" in n.attrs
    ]
    assert checked == [("family", "qwen"), ("quant_bucket", "Q4")]


@pytest.mark.parametrize("url", ["/", FILTERED])
def test_param_bounds_are_decimal_number_fields_that_still_auto_refresh(
    client: TestClient, url: str
) -> None:
    form, _ = _filters(url, client)
    for name in ("min_params_b", "max_params_b"):
        field = _by_id(form, name)
        assert field.attrs["name"] == name
        assert field.attrs["type"] == "number"
        assert field.attrs["inputmode"] == "decimal"
        assert field.attrs["min"] == "0"
        assert field.attrs["step"] == "0.1"
        assert "js-auto-filter-text" in field.classes()
    boxes = [n for n in form.walk() if n.attrs.get("type") == "checkbox"]
    assert boxes and all("js-auto-filter" in n.classes() for n in boxes)


def test_apply_filters_button_sits_in_a_noscript_inside_the_disclosure(
    client: TestClient,
) -> None:
    _, filters = _filters("/", client)
    noscripts = [n for n in filters.walk() if n.tag == "noscript"]
    assert len(noscripts) == 1
    assert 'type="submit" class="btn-secondary">Apply filters</button>' in client.get("/").text


def test_filters_have_no_legacy_sidebar_markup(client: TestClient) -> None:
    html = client.get(FILTERED).text
    for gone in ("filters-sidebar", "filters-disclosure", "field-help", 'class="checkbox"'):
        assert gone not in html, gone


def test_no_script_opens_the_filters() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert "filters" in js  # the count listener is there
    assert not re.search(r"\.open\s*=(?!=)", js), "a script must not set details.open"
    assert 'setAttribute("open"' not in js and "setAttribute('open'" not in js
    assert "data-open-wide" not in (TEMPLATES / "index.html").read_text(encoding="utf-8")


def test_filter_count_listener_is_delegated_on_document() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert 'document.addEventListener("input", updateFilterCount)' in js
    assert 'document.addEventListener("change", updateFilterCount)' in js
    assert "[data-fcount]" in js
    assert "input[type=checkbox]:checked" in js
    assert "input[type=number]" in js


def test_filter_checkboxes_have_a_44px_target_at_phone_width() -> None:
    css = _css()
    assert "min-height: var(--tap)" in _rules(css, ".choice")[0]
    assert "--tap: 44px" in css
    box = _rules(css, ".choice input")[0]
    assert "width: 22px" in box and "height: 22px" in box
    assert "min-height: 52px" in _rules(css, ".filters > summary")[0]
    assert "display: none" in _rules(css, ".filters > summary::-webkit-details-marker")[0]


def test_legacy_filter_rules_are_gone() -> None:
    css = _css()
    for gone in (".filters-sidebar", ".filters-disclosure", ".filter-group legend", ".checkbox"):
        assert gone not in css, gone


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
    assert main.ASSET_VERSION > "20261025f"
    assert f"styles.css?v={main.ASSET_VERSION}" in client.get("/").text


# ---- T-013: the answer tile ----

REFERENCE = "/results?vram_gb=24&context_tokens=4096"
HX = {"HX-Request": "true"}


def _answer(html: str) -> _Node:
    return next(n for n in _tree(html).walk() if n.attrs.get("id") == "summary-banner")


def _shell_children(html: str) -> list[_Node]:
    shell = next(n for n in _tree(html).walk() if n.attrs.get("id") == "results-shell")
    out: list[_Node] = []
    for wrapper in shell.children:
        out.extend(wrapper.children)
    return out


def test_answer_tile_is_the_first_content_of_the_results_shell(client: TestClient) -> None:
    html = client.get(REFERENCE).text
    first = _shell_children(html)[0]
    assert first.attrs["id"] == "summary-banner"
    assert first.tag == "section" and {"answer", "pane"} <= set(first.classes())
    assert first.attrs["aria-labelledby"] == "ans-h"
    headline = first.children[0]
    assert (headline.tag, headline.attrs["id"]) == ("h2", "ans-h")
    assert headline.all_text() == "71 models fit or are tight"
    # Top Picks and the cards come after the tile.
    assert html.index('id="summary-banner"') < html.index("Top Picks for you")
    assert html.index("Top Picks for you") < html.index('id="results-cards"')


def test_headline_counts_model_cards_and_keeps_the_singular(client: TestClient) -> None:
    one = _answer(client.get("/results?vram_gb=3&family=phi").text)
    assert one.children[0].all_text() == "1 model fit or are tight"
    none = _answer(client.get("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8").text)
    assert none.children[0].all_text() == "0 models fit or are tight"


def test_answer_tile_holds_the_context_line_counts_and_note(client: TestClient) -> None:
    answer = _answer(client.get(REFERENCE).text)
    ctx = next(n for n in answer.walk() if "ctx-line" in n.classes())
    spans = [" ".join((b.all_text(), s.text.strip())) for s in ctx.children for b in s.children]
    assert spans == ["Hardware: Manual VRAM (24 GB)", "Context: 4k"]
    counts = next(n for n in answer.walk() if "counts" in n.classes())
    items = [c for c in counts.children if c.tag == "li"]
    labels = [next(n for n in li.children if n.tag == "span").all_text() for li in items]
    assert labels == ["Fits", "Tight", "Won't fit"]
    # Each count carries a distinct shape cue, not only a colour.
    symbols = [next(n for n in li.walk() if n.tag == "use").attrs["href"] for li in items]
    assert symbols == ["#i-fits", "#i-tight", "#i-wont"]
    assert [next(n for n in li.children if n.tag == "b").all_text() for li in items][0] == "71"
    note = next(n for n in answer.walk() if "note" in n.classes())
    assert note.all_text() == (
        "Only Fits and Tight models are listed. "
        "Pick a quantisation on any card to see how it changes."
    )
    assert note.attrs["title"] == (
        "Fits leaves reserve headroom. Tight can run, but is close to your VRAM limit."
    )
    assert "assumed 8 GB baseline" not in answer.all_text()


def test_empty_state_note_keeps_only_the_first_sentence(client: TestClient) -> None:
    answer = _answer(client.get("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8").text)
    note = next(n for n in answer.walk() if "note" in n.classes())
    assert note.all_text() == "Only Fits and Tight models are listed."
    assert note.attrs["title"].startswith("Fits leaves reserve headroom.")


def test_sprite_symbols_for_the_count_icons_are_in_the_page(client: TestClient) -> None:
    html = client.get(REFERENCE).text
    for symbol in ("i-fits", "i-tight", "i-wont", "i-info"):
        assert f'<symbol id="{symbol}"' in html


def test_default_8gb_note_sits_in_the_tile_under_the_context_line(client: TestClient) -> None:
    answer = _answer(client.get("/").text)
    notice = next(n for n in answer.walk() if "notice" in n.classes())
    assert notice.all_text() == (
        "No GPU or VRAM selected yet. Showing estimates with an assumed 8 GB baseline."
    )
    assert notice.attrs["role"] == "status"
    kids = [c.classes()[0] if c.classes() else c.tag for c in answer.children]
    assert kids.index("ctx-line") < kids.index("notice") < kids.index("counts")


def test_messages_render_in_the_tile_between_the_context_line_and_the_counts(
    client: TestClient,
) -> None:
    html = client.get("/results?vram_gb=abc").text
    answer = _answer(html)
    alert = next(n for n in answer.walk() if n.attrs.get("role") == "alert")
    assert alert.all_text().startswith("Please fix the following input issues:")
    assert [li.all_text() for li in alert.walk() if li.tag == "li"] == [
        "Invalid value for vram_gb."
    ]
    kids = [c.classes()[0] if c.classes() else c.tag for c in answer.children]
    assert kids.index("ctx-line") < kids.index("notice") < kids.index("counts")

    html = client.get("/results?gpu_search=RTX").text
    answer = _answer(html)
    status = [n for n in answer.walk() if n.attrs.get("role") == "status"]
    assert len(status) == 1 and "matched multiple entries" in status[0].all_text()
    html = client.get("/results?gpu_search=RTX 3060 12GB").text
    status = [n for n in _answer(html).walk() if n.attrs.get("role") == "status"]
    assert len(status) == 1 and "was not found. Using 12 GB inferred" in status[0].all_text()


def test_an_in_place_refresh_shows_only_the_current_querys_messages(client: TestClient) -> None:
    bad = client.get("/results?vram_gb=abc", headers=HX).text
    assert bad.count('role="alert"') == 1
    good = client.get("/results?vram_gb=24", headers=HX).text
    assert 'role="alert"' not in good
    assert "Invalid value" not in good
    partial = client.get("/results/partials/content?vram_gb=24", headers=HX).text
    assert 'role="alert"' not in partial
    # A refresh that fixes a GPU search also drops its message.
    assert "matched multiple" not in client.get("/results?vram_gb=24&gpu_search=", headers=HX).text


def test_active_filter_chips_are_links_inside_the_tile(client: TestClient) -> None:
    answer = _answer(client.get(FILTERED).text)
    block = next(n for n in answer.walk() if "active-filters" in n.classes())
    assert block.children[0].all_text() == "Active filters:"
    chips = [n for n in block.children if n.tag == "a"]
    assert [c.all_text() for c in chips] == ["Family: qwen ×", "Quant: Q4 ×"]
    assert all(c.attrs["href"].startswith("/results") for c in chips)
    assert not [n for n in _answer(client.get(REFERENCE).text).walk() if n.tag == "a"]
    assert "Active filters:" not in client.get(REFERENCE).text


def test_old_headings_and_sentences_are_gone(client: TestClient) -> None:
    for url in (REFERENCE, "/", "/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8"):
        html = client.get(url).text
        assert "Summary for this query" not in html
        assert "Showing only Fits and Tight models" not in html
        assert "results-header" not in html
        assert "fit or are tight." not in html


def test_results_heading_is_visually_hidden_above_the_cards(client: TestClient) -> None:
    html = client.get(REFERENCE).text
    heading = next(n for n in _tree(html).walk() if n.tag == "h2" and n.all_text() == "Results")
    assert heading.classes() == ["sr-only"]
    assert html.index(">Results</h2>") < html.index('id="results-cards"')
    assert "position: absolute" in _rules(_css(), ".sr-only")[0]
    assert html.count(">Results</h2>") == 1


def test_frozen_ids_and_partial_endpoints_still_render_their_templates(
    client: TestClient,
) -> None:
    html = client.get(REFERENCE).text
    for ident in (
        "results-shell",
        "results-summary-shell",
        "results-list-shell",
        "summary-banner",
        "results-cards",
    ):
        assert html.count(f'id="{ident}"') == 1, ident
    summary = client.get("/results/partials/summary?vram_gb=24").text
    assert 'id="summary-banner"' in summary and 'id="results-cards"' not in summary
    listing = client.get("/results/partials/list?vram_gb=24").text
    assert 'id="results-cards"' in listing and 'id="summary-banner"' not in listing
    content = client.get("/results/partials/content?vram_gb=24", headers=HX).text
    assert 'id="results-summary-shell"' in content and 'id="results-list-shell"' in content


def test_no_picks_layout_is_keyed_on_markup_inside_the_results_shell(client: TestClient) -> None:
    with_picks = _answer(client.get(REFERENCE, headers=HX).text)
    assert "no-picks" not in with_picks.classes()
    without = _answer(client.get("/results?vram_gb=2&context_tokens=4096", headers=HX).text)
    assert "no-picks" in without.classes()
    assert client.get("/results?vram_gb=2&context_tokens=4096").text.count("no-picks") == 1
    main_node = _one(_tree(client.get("/results?vram_gb=2").text), "main")
    assert main_node.classes() == ["bento"]


def test_answer_tile_css_matches_the_mockup_tokens() -> None:
    css = _css()
    assert "var(--fs-display)" in _rules(css, ".headline")[0]
    assert "repeat(3, 1fr)" in _rules(css, ".counts")[0]
    assert "var(--tap)" in _rules(css, ".chip-link")[0]
    assert "var(--wont-bg)" in _rules(css, ".notice.is-alert")[0]
    assert "grid-row: 1 / 3" in _rules(css, ".answer.no-picks")[0]
    for gone in (".summary-banner", ".summary-grid", ".summary-stat", ".results-header"):
        assert gone not in css, gone


# ---- T-014: Top Picks tiles ----


def _picks(html: str) -> _Node:
    found = [n for n in _tree(html).walk() if "picks" in n.classes()]
    assert len(found) == 1
    return found[0]


def test_top_picks_is_a_closed_disclosure_after_the_tile_with_an_opening_script(
    client: TestClient,
) -> None:
    html = client.get(REFERENCE).text
    picks = _picks(html)
    assert picks.tag == "details" and "open" not in picks.attrs
    assert "data-open-wide" in picks.attrs
    assert {"pane", "lite"} <= set(picks.classes())
    summary = picks.children[0]
    assert summary.tag == "summary" and summary.all_text() == "Top Picks for you"
    # Shell order: answer tile, Top Picks, then the cards.
    kids = _shell_children(html)
    assert [k.attrs.get("id") or k.tag for k in kids[:3]] == ["summary-banner", "details", "script"]
    assert (
        html.index('id="summary-banner"') < html.index('class="picks') < html.index("results-cards")
    )
    # The script follows the details directly and opens it from 900 px.
    after = html[html.index("</details>", html.index('class="picks')) :]
    script = re.match(r"</details>\s*<script>(.*?)</script>", after, re.S)
    assert script, "inline script must follow the details directly"
    assert "matchMedia('(min-width: 900px)')" in script.group(1)
    assert "data-open-wide" in script.group(1) and ".open = true" in script.group(1)


def test_top_picks_script_also_ships_in_an_in_place_refresh(client: TestClient) -> None:
    html = client.get(REFERENCE, headers=HX).text
    assert html.count("data-open-wide") >= 2  # the attribute and the script that reads it
    assert "(min-width: 900px)" in html


def test_top_picks_tiles_carry_the_new_descriptions_and_no_ranking_words(
    client: TestClient,
) -> None:
    picks = _picks(client.get(REFERENCE).text)
    tiles = [n for n in picks.walk() if n.tag == "article"]
    assert len(tiles) == 3 and all("pick" in t.classes() for t in tiles)

    def part(tile: _Node, cls: str) -> _Node:
        return next(n for n in tile.walk() if cls in n.classes())

    assert [part(t, "pick-title").all_text() for t in tiles] == [
        "Best quality that fits",
        "Best balance",
        "Best small",
    ]
    assert [part(t, "pick-desc").all_text() for t in tiles] == [
        "The strongest model that still leaves room to spare.",
        "Nearly as capable, and comfortable to run.",
        "The lightest model that still runs on your GPU.",
    ]
    for tile in tiles:
        assert re.fullmatch(r"\d+\.\dB · \w+", part(tile, "pick-meta").all_text())
        link = next(n for n in tile.walk() if n.tag == "a")
        assert link.attrs["href"].startswith("/models/model_")
        assert "vram_gb=24" in link.attrs["href"] and "context_tokens=4096" in link.attrs["href"]
    text = picks.all_text().lower()
    assert "ranked" not in text and "ranking" not in text
    source = (ROOT / "src/vramsherpa/main.py").read_text(encoding="utf-8")
    assert "ranked" not in source.split("def _top_picks")[1].split("def _example_chips")[0]


def test_top_picks_are_absent_when_r024_says_so(client: TestClient) -> None:
    html = client.get("/results?vram_gb=2&context_tokens=4096").text
    assert "Top Picks for you" not in html and "data-open-wide" not in html
    assert 'class="picks' not in html


def test_top_picks_css_matches_the_mockup_and_reuses_the_one_keyframes() -> None:
    css = _css()
    assert css.count("@keyframes whyIn") == 1
    assert _rules(css, "details[open] > .pick-grid")[0].startswith(" animation: whyIn")
    assert "repeat(3, minmax(0, 1fr))" in css
    assert "var(--inset)" in _rules(css, ".pick")[0]
    assert "min-height: 44px" in _rules(css, ".picks > summary")[0]
    for gone in (".top-picks", ".top-pick-grid", ".top-pick-card", ".eyebrow"):
        assert gone not in css, gone
