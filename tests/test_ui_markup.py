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


# ---- T-015: model cards, the quantisation switcher and the fit gauge ----

GATED_QUERY = "/results?vram_gb=16&context_tokens=4096"
COMPONENTS = TEMPLATES / "components.html"


def _cards(html: str) -> list[_Node]:
    return [n for n in _tree(html).walk() if "data-model-card" in n.attrs]


def _render_macro(call: str, **context: str) -> str:
    template = main.templates.env.from_string(
        '{% from "components.html" import verdict, gated_tag %}' + call
    )
    return template.render(**context)


def test_verdict_and_gated_marker_are_macros_imported_by_the_results_list() -> None:
    components = COMPONENTS.read_text(encoding="utf-8")
    assert "{% macro verdict(" in components and "{% macro gated_tag(" in components
    listing = (TEMPLATES / "results_list.html").read_text(encoding="utf-8")
    assert 'from "components.html" import' in listing
    assert "verdict(" in listing and "gated_tag(" in listing
    # The class names live in the macro file alone, so a page cannot restyle them on its own.
    assert 'class="state' not in listing and 'class="tag-gated' not in listing


@pytest.mark.parametrize(
    ("label", "key"), [("Fits", "fits"), ("Tight", "tight"), ("Won't fit", "wont")]
)
def test_verdict_macro_pairs_the_label_with_one_shape(label: str, key: str) -> None:
    node = _tree(_render_macro("{{ verdict(label) }}", label=label)).children[0]
    assert node.classes() == ["state", f"state-{key}"]
    icon = next(n for n in node.walk() if n.tag == "svg")
    assert icon.attrs["aria-hidden"] == "true"
    assert next(n for n in icon.walk() if n.tag == "use").attrs["href"] == f"#i-{key}"
    assert node.all_text() == label


def test_gated_macro_has_a_lock_icon_and_the_licence_tooltip() -> None:
    node = _tree(_render_macro("{{ gated_tag() }}")).children[0]
    assert node.classes() == ["tag-gated"]
    assert node.attrs["title"] == (
        "Accept the model licence on Hugging Face before downloading the weights."
    )
    assert next(n for n in node.walk() if n.tag == "use").attrs["href"] == "#i-lock"
    assert node.all_text() == "Gated"


def test_sprite_carries_every_icon_the_cards_use(client: TestClient) -> None:
    html = client.get(GATED_QUERY).text
    for symbol in ("i-fits", "i-tight", "i-wont", "i-lock"):
        assert html.count(f'id="{symbol}"') == 1, symbol


def test_cards_sit_in_one_glass_pane_as_a_grid_of_inset_cards(client: TestClient) -> None:
    html = client.get(GATED_QUERY).text
    section = next(n for n in _tree(html).walk() if n.attrs.get("id") == "results-cards")
    assert section.tag == "section" and {"results", "pane", "lite"} <= set(section.classes())
    grid = next(n for n in section.children if n.tag == "div")
    assert grid.classes() == ["card-grid"]
    cards = [n for n in grid.children if n.tag == "article"]
    assert cards and all({"model-card", "inset"} <= set(c.classes()) for c in cards)
    assert len(cards) == len(_cards(html))


def test_every_pane_has_one_gauge_and_the_verdict_before_the_margin(client: TestClient) -> None:
    html = client.get(GATED_QUERY).text
    cards = _cards(html)
    assert cards
    for card in cards:
        panes = [n for n in card.walk() if "data-variant-pane" in n.attrs]
        assert len(panes) == 4
        for pane in panes:
            gauges = [n for n in pane.walk() if n.attrs.get("role") == "img"]
            assert len(gauges) == 1
            assert gauges[0].classes() == ["track"]
            assert gauges[0].attrs["aria-label"].startswith("Needs ")
            top = next(n for n in pane.walk() if "fit-top" in n.classes())
            assert [c.classes()[0] for c in top.children] == ["state", "headroom"]
            captions = next(n for n in pane.walk() if "scale" in n.classes())
            assert [
                c.all_text().endswith("GB") or "GB needed" in c.all_text()
                for c in captions.children
            ] == [True, True]
            assert [
                n for n in pane.walk() if n.tag == "svg" and n.attrs.get("aria-hidden") != "true"
            ] == []


def test_bar_class_follows_the_verdict(client: TestClient) -> None:
    html = client.get(GATED_QUERY).text
    seen = set()
    for card in _cards(html):
        for pane in (n for n in card.walk() if "data-variant-pane" in n.attrs):
            state = next(n for n in pane.walk() if "state" in n.classes()).classes()[1]
            bar = next(n for n in pane.walk() if "bar" in n.classes()).classes()[1]
            assert bar == state.replace("state-", "bar-")
            seen.add(state)
    assert seen == {"state-fits", "state-tight", "state-wont"}


def test_switcher_is_one_row_of_four_with_a_thumb_hook_and_struck_wont_fit(
    client: TestClient,
) -> None:
    html = client.get(GATED_QUERY).text
    for card in _cards(html):
        group = next(n for n in card.walk() if "quants" in n.classes())
        assert "data-thumb" in group.attrs and group.attrs["role"] == "group"
        buttons = [n for n in group.children if n.tag == "button"]
        assert [b.all_text() for b in buttons] == ["Q4", "Q5", "Q8", "FP16"]
        assert [b.attrs["aria-pressed"] for b in buttons].count("true") == 1
        panes = {
            n.attrs["data-variant-pane"]: next(s for s in n.walk() if "state" in s.classes())
            for n in card.walk()
            if "data-variant-pane" in n.attrs
        }
        for button in buttons:
            wont = "state-wont" in panes[button.attrs["data-variant"]].classes()
            assert ("data-oversize" in button.attrs) == wont
            if wont:
                assert button.attrs["title"] == "Does not fit in 16 GB"


def test_gated_models_carry_the_marker_inside_the_heading(client: TestClient) -> None:
    html = client.get(GATED_QUERY).text
    gated = [c for c in _cards(html) if any("tag-gated" in n.classes() for n in c.walk())]
    assert gated
    for card in gated:
        heading = next(n for n in card.walk() if n.tag == "h3")
        assert any("tag-gated" in n.classes() for n in heading.walk())


def test_card_meta_keeps_params_and_family(client: TestClient) -> None:
    html = client.get(GATED_QUERY).text
    meta = next(n for n in _tree(html).walk() if "model-meta" in n.classes())
    spans = [n for n in meta.children if n.tag == "span"]
    assert re.fullmatch(r"\d+(\.\d+)?B params", spans[0].all_text())
    assert spans[1].classes() == ["fam"] and spans[1].all_text()


def test_card_css_shapes_reserve_and_strike_through() -> None:
    css = _css()
    assert "dashed" in _rules(css, ".resv")[0]
    assert "background" not in _rules(css, ".resv")[0]
    oversize = _rules(css, ".quants button[data-oversize]")[0]
    assert "line-through" in oversize
    quants = _rules(css, ".quants")[0]
    assert "repeat(4, minmax(0, 1fr))" in quants and "width: 100%" in quants
    assert any("overflow-wrap: anywhere" in r for r in _rules(css, ".model-card h3"))
    for key in ("fits", "tight", "wont"):
        assert f"--pill-bg: var(--{key}-bg)" in _rules(css, f".state-{key}")[0]
        assert f"var(--{key}-bar)" in _rules(css, f".bar-{key}")[0]


def test_verdict_and_margin_are_no_smaller_than_other_card_text() -> None:
    css = _css()
    assert "var(--fs-lead)" in _rules(css, ".state")[0]
    assert "var(--fs-lead)" in _rules(css, ".headroom")[0]
    assert "34px" in _rules(css, ".headroom b")[0]
    # Everything else in a pane (captions, breakdown link) is the caption or small size.
    assert "var(--fs-caption)" in _rules(css, ".scale")[0]


def test_old_card_rules_are_gone() -> None:
    css = _css()
    for dead in (".model-list", ".state-wont-fit", ".bar-wont-fit", ".state::before", ".tag {"):
        assert dead not in css, dead


def test_app_js_thumbs_cover_the_switcher_and_run_after_swaps() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert '[aria-pressed="true"]' in js
    assert "function addThumbs" in js
    assert 'addEventListener("htmx:afterSettle", refreshThumbs)' in js
    assert 'addEventListener("htmx:historyRestore", refreshThumbs)' in js


def test_app_js_motion_is_progressive_and_off_under_reduced_motion() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in js
    assert "is-entering" in js and "htmx-added" in js
    assert "!prefersReducedMotion()" in js
    # The switcher still works with no motion: visibility and aria-pressed come first.
    assert "pane.hidden = !show" in js
    # Sheen follows a fine pointer through one delegated listener.
    assert "(hover: hover) and (pointer: fine)" in js
    assert 'document.addEventListener("pointermove"' in js
    css = _css()
    assert "transition: none !important" in css
    assert "@keyframes paneIn" in css


def test_asset_version_moved_on_for_the_card_restyle() -> None:
    assert main.ASSET_VERSION > "20261025h"


# ---- T-016: How this is estimated ----

WHY_LABELS = [
    "Model weights",
    "Context memory (KV cache)",
    "Runtime overhead",
    "Total needed",
    "Safety reserve Kept free, not part of the total",
]


def _first_pane(html: str) -> _Node:
    return next(n for n in _tree(html).walk() if "data-variant-pane" in n.attrs)


def _why_summary(pane: _Node) -> _Node:
    return next(n for n in pane.walk() if n.tag == "summary")


def test_breakdown_is_a_definition_list_with_the_total_ruled_off_before_the_reserve(
    client: TestClient,
) -> None:
    url = _why_summary(_first_pane(client.get(REFERENCE).text)).attrs["hx-get"]
    root = _tree(client.get(url).text)
    title = next(n for n in root.walk() if "why-title" in n.classes())
    assert re.fullmatch(r"\S+ on .+", title.all_text())
    lists = [n for n in root.walk() if n.tag == "dl"]
    assert len(lists) == 1 and "why-list" in lists[0].classes()
    rows = [n for n in lists[0].children if n.tag == "div"]
    assert [next(c for c in r.children if c.tag == "dt").all_text() for r in rows] == WHY_LABELS
    for row in rows:
        dd = next(c for c in row.children if c.tag == "dd")
        assert re.fullmatch(r"\d+\.\d\d GB", dd.all_text())
    assert [r.classes() for r in rows] == [[], [], [], ["total"], ["reserve"]]
    assert not any(n.tag == "ul" for n in root.walk())


def test_total_needed_equals_the_gb_needed_caption_of_the_same_pane(client: TestClient) -> None:
    html = client.get(REFERENCE).text
    checked = 0
    for pane in (n for n in _tree(html).walk() if "data-variant-pane" in n.attrs):
        caption = next(n for n in pane.walk() if "scale" in n.classes())
        needed = caption.children[0].all_text()
        assert needed.endswith(" GB needed")
        root = _tree(client.get(_why_summary(pane).attrs["hx-get"]).text)
        total = next(n for n in root.walk() if "total" in n.classes())
        assert next(c for c in total.children if c.tag == "dd").all_text() == needed.replace(
            " needed", ""
        )
        checked += 1
        if checked == 6:
            break
    assert checked > 1


def test_breakdown_stays_collapsed_and_fetched_once_with_its_placeholder(
    client: TestClient,
) -> None:
    pane = _first_pane(client.get(REFERENCE).text)
    panel = next(n for n in pane.walk() if "why-panel" in n.classes())
    assert panel.tag == "details" and "open" not in panel.attrs
    summary = _why_summary(pane)
    assert summary.all_text() == "How this is estimated"
    assert summary.attrs["hx-trigger"] == "click once"
    assert summary.attrs["hx-swap"] == "innerHTML"
    content = next(n for n in panel.walk() if "why-content" in n.classes())
    assert summary.attrs["hx-target"] == "#" + content.attrs["id"]
    idle = next(n for n in content.walk() if "why-idle" in n.classes())
    assert idle.all_text() == "Open to load the breakdown."


def test_loading_state_is_five_shimmer_rows_in_markup_and_shown_by_htmx_request(
    client: TestClient,
) -> None:
    pane = _first_pane(client.get(REFERENCE).text)
    loading = next(n for n in pane.walk() if "why-loading" in n.classes())
    assert "Loading the breakdown" in loading.all_text()
    rows = [n for n in loading.walk() if n.tag == "i"]
    assert len(rows) == 5
    css = _css()
    assert "display: none" in _rules(css, ".why-loading")[0]
    shown = ".why-panel > summary.htmx-request + .why-content .why-loading"
    assert "display: block" in _rules(css, shown)[0]
    assert (
        "display: none"
        in _rules(css, ".why-panel > summary.htmx-request + .why-content .why-idle")[0]
    )
    assert "animation: shimmer" in _rules(css, ".sk-list i")[0]
    # Under reduced motion the global rule stops the shimmer; the rows stay as flat bars.
    assert re.search(
        r"prefers-reduced-motion: reduce\) \{ \.sk-list i \{ background: var\(--sk-a\)", css
    )
    js = (ROOT / "src/vramsherpa/static/js/app.js").read_text(encoding="utf-8")
    assert "Loading the breakdown" not in js


def test_breakdown_css_reuses_the_one_keyframes_and_fits_a_phone() -> None:
    css = _css()
    assert css.count("@keyframes whyIn") == 1 and css.count("@keyframes shimmer") == 1
    # Two fades, as in the spec sheet: the panel fades in on opening, then the rows fade in when
    # they load. htmx swaps in a fresh .why-list once per card, so an animation on .why-list is
    # the app's counterpart of the mockup's `.why-content.is-loaded .why-list` hook.
    assert "animation: whyIn" in _rules(css, "details[open] > .why-content")[0]
    assert "animation: whyIn var(--dur-2) var(--ease-out)" in _rules(css, ".why-list")[0]
    row = _rules(css, ".why-list > div")[0]
    assert "justify-content: space-between" in row and "gap: 16px" in row
    assert "min-width: 0" in _rules(css, ".why-list dt")[0]
    assert "margin: 0" in _rules(css, ".why-list dd")[0]  # the browser default indents dd
    assert "dashed" in _rules(css, ".why-list .reserve")[0]
    assert "2px solid" in _rules(css, ".why-list .total")[0]
    assert ".why-breakdown ul" not in css


def test_breakdown_margins_outrank_the_model_card_reset() -> None:
    # `.model-card p` (0,1,1) beats a bare `.why-title` (0,1,0); the panel prefix (0,2,0) wins.
    css = _css()
    assert "margin: 0" in _rules(css, ".model-card h3, .model-card p")[0]
    assert "margin: 4px 0 8px" in _rules(css, ".why-panel .why-title")[0]
    assert "margin: 4px 0 0" in _rules(css, ".why-panel .sk-note")[0]


def test_breakdown_pseudo_elements_are_border_box_like_the_mockup() -> None:
    # The app reset is `* { box-sizing: border-box }`, which skips ::before; the mockups use
    # `*, *::before, *::after`. Without border-box the 9 px chevron draws 11 px and the 14 px
    # spinner 18 px, and the summary label shifts right by 2 px.
    css = _css()
    reset_covers_pseudo = bool(re.search(r"\*,\s*\*::before,\s*\*::after\s*\{[^}]*border-box", css))
    for selector, width in (
        (".why-panel > summary::before", "width: 9px"),
        (".why-panel .sk-note::before", "width: 14px"),
    ):
        rule = _rules(css, selector)[0]
        assert width in rule, selector
        assert reset_covers_pseudo or "box-sizing: border-box" in rule, selector


def test_breakdown_fragment_has_no_unstyled_wrapper(client: TestClient) -> None:
    url = _why_summary(_first_pane(client.get(REFERENCE).text)).attrs["hx-get"]
    html = client.get(url).text
    assert "why-breakdown" not in html
    assert html.lstrip().startswith('<p class="why-title">')


def test_asset_version_moved_on_for_the_breakdown() -> None:
    assert main.ASSET_VERSION > "20261025l"


# ---- empty and unseeded states (T-017) ----


def _empty(html: str) -> _Node:
    return next(n for n in _tree(html).walk() if "empty" in n.classes())


def _render_list(**context: object) -> str:
    base = {
        "model_cards": [],
        "catalog_seeded": True,
        "empty_state_q4_url": None,
        "empty_state_lower_context_url": None,
        "summary_counts": {"Fits": 0, "Tight": 0, "Won't fit": 0},
    }
    return main.templates.env.get_template("results_list.html").render(**{**base, **context})


def _wont_line(count: int) -> list[str]:
    html = _render_list(summary_counts={"Fits": 0, "Tight": 0, "Won't fit": count})
    return [
        n.all_text() for n in _empty(html).walk() if n.tag == "p" and "Won't fit" in n.all_text()
    ]


def test_empty_state_counts_models_not_variants() -> None:
    assert _wont_line(95) == ["There are 95 models in Won't fit, hidden by default."]
    assert _wont_line(2) == ["There are 2 models in Won't fit, hidden by default."]
    assert _wont_line(1) == ["There is 1 model in Won't fit, hidden by default."]
    assert _wont_line(0) == []
    for count in (0, 1, 2, 95):
        html = _render_list(summary_counts={"Fits": 0, "Tight": 0, "Won't fit": count})
        assert "variants" not in _empty(html).all_text()


def test_empty_state_live_page_says_models(client: TestClient) -> None:
    html = client.get("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8").text
    empty = _empty(html)
    paras = [n.all_text() for n in empty.children if n.tag == "p"]
    assert paras[:2] == ["No models currently fit this setup.", "Try one of these next steps:"]
    assert re.fullmatch(r"There are \d+ models in Won't fit, hidden by default\.", paras[2])
    assert all("variants" not in p for p in paras)


def test_empty_state_sits_in_the_results_pane_as_an_inset_sheet(client: TestClient) -> None:
    html = client.get("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8").text
    empty = _empty(html)
    assert "inset" in empty.classes()
    pane = empty.parent
    assert pane is not None and pane.tag == "section"
    assert {"results", "pane", "lite"} <= set(pane.classes())
    assert pane.attrs["aria-label"] == "Models"
    title = next(n for n in empty.children if "empty-title" in n.classes())
    assert title.all_text() == "No models currently fit this setup."


@pytest.mark.parametrize(
    ("url", "kinds"),
    [
        ("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8", ("a", "a")),
        ("/results?vram_gb=1&context_tokens=2048&quant_bucket=Q4", ("span", "span")),
        ("/results?vram_gb=1&context_tokens=4096&quant_bucket=Q4", ("span", "a")),
    ],
)
def test_empty_state_next_steps_are_links_or_inert_text(
    client: TestClient, url: str, kinds: tuple[str, str]
) -> None:
    actions = next(n for n in _empty(client.get(url).text).walk() if "empty-actions" in n.classes())
    chips = actions.children
    assert [c.all_text() for c in chips] == ["Try Q4", "Lower context"]
    assert tuple(c.tag for c in chips) == kinds
    for chip in chips:
        assert "action-chip" in chip.classes()
        if chip.tag == "a":
            assert chip.attrs["href"].startswith("/results?")
        else:
            # Inert text: no href, no tabindex, no role, so it is not focusable.
            assert set(chip.attrs) == {"class"}


def test_unseeded_state_is_a_status_in_the_results_pane(app, db_session: Session) -> None:
    html = TestClient(app).get("/").text
    empty = _empty(html)
    status = next(n for n in empty.walk() if n.attrs.get("role") == "status")
    assert [p.all_text() for p in status.children if p.tag == "p"] == [
        "Catalog is not initialized yet.",
        "Seed the catalog data, then refresh this page.",
    ]
    assert any(n.tag == "use" and n.attrs["href"] == "#i-info" for n in status.walk())
    assert "No models currently fit this setup." not in html
    assert "action-chip" not in html
    assert empty.parent is not None and "results" in empty.parent.classes()


def test_empty_state_styles_follow_the_mockup() -> None:
    css = _css()
    assert "padding: 28px 24px 30px" in _rules(css, ".empty")[0]
    assert "font-size: 28px" in _rules(css, ".empty .empty-title")[0]
    assert "var(--on-btn)" in _rules(css, "a.action-chip")[0]
    assert "dashed" in _rules(css, "span.action-chip")[0]
    assert "empty-state-callout" not in css and "empty-state-actions" not in css


def test_asset_version_moved_on_for_the_empty_states() -> None:
    assert main.ASSET_VERSION > "20261025m"


# ---- T-018: model detail ----

GATED_DETAIL = "/models/model_llama_31_8b_instruct?vram_gb=24&context_tokens=2048"
OPEN_DETAIL = "/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096"
ERROR_DETAIL = "/models/model_qwen25_7b_instruct?vram_gb=abc"
VARIANT_HEADINGS = [
    "Variant",
    "Bits",
    "Required VRAM (GB)",
    "Available VRAM (GB)",
    "Margin (GB)",
    "Status",
    "Provenance",
]


def _by_class(root: _Node, cls: str) -> list[_Node]:
    return [n for n in root.walk() if cls in n.classes()]


def test_detail_facts_are_four_labelled_tiles_with_the_same_values(client: TestClient) -> None:
    root = _tree(client.get(GATED_DETAIL).text)
    facts = _by_class(root, "facts")[0]
    assert facts.tag == "dl"
    tiles = [[c.all_text() for c in tile.children] for tile in facts.children]
    assert tiles == [
        ["Family", "llama"],
        ["Params", "8.0B"],
        ["Type", "instruct"],
        ["License", "Llama 3.1 Community License"],
    ]
    assert "KV cache estimate per 1k context: 0.480 GB" in root.all_text()
    assert [
        n.all_text() for n in root.walk() if n.tag == "h2" and "sources" in n.parent.classes()
    ] == ["Sources"]


def test_detail_headings_for_the_form_and_the_table_are_visually_hidden(
    client: TestClient,
) -> None:
    root = _tree(client.get(OPEN_DETAIL).text)
    hidden = {n.all_text() for n in root.walk() if n.tag == "h2" and "sr-only" in n.classes()}
    assert {"Recompute for other hardware", "Variants"} <= hidden
    assert "clip: rect(0 0 0 0)" in _rules(_css(), ".sr-only")[0]


def test_detail_gated_marker_and_verdicts_come_from_the_shared_macros(client: TestClient) -> None:
    gated = _tree(client.get(GATED_DETAIL).text)
    h1 = _one(gated, "h1")
    tag = next(n for n in h1.walk() if "tag-gated" in n.classes())
    assert tag.all_text() == "gated"
    assert tag.attrs["title"].startswith("Gated repo: you must accept the model license")
    assert next(n for n in tag.walk() if n.tag == "use").attrs["href"] == "#i-lock"
    assert all(n.all_text() != "Gated" for n in gated.walk() if "tag-gated" in n.classes())
    assert "Weights are gated on Hugging Face" in gated.all_text()
    verdicts = _by_class(gated, "state")
    assert len(verdicts) == 4
    assert all("state-" in " ".join(n.classes()) for n in verdicts)
    assert not [n for n in gated.walk() if any(c.startswith("badge") for c in n.classes())]

    open_page = _tree(client.get(OPEN_DETAIL).text)
    assert not _by_class(open_page, "tag-gated")
    assert "Weights are gated on Hugging Face" not in open_page.all_text()


def test_gated_macro_label_defaults_to_the_card_word() -> None:
    assert _tree(_render_macro("{{ gated_tag() }}")).children[0].all_text() == "Gated"
    assert _tree(_render_macro('{{ gated_tag("gated") }}')).children[0].all_text() == "gated"


def test_detail_form_keeps_its_fields_and_the_primary_button(client: TestClient) -> None:
    form = _one(_tree(client.get(OPEN_DETAIL).text), "form")
    assert form.attrs["method"] == "get"
    fields = {n.attrs.get("name"): n for n in form.walk() if n.tag in {"select", "input"}}
    assert set(fields) == {"gpu_id", "vram_gb", "context_tokens"}
    assert fields["vram_gb"].attrs["inputmode"] == "decimal"
    assert fields["vram_gb"].attrs["value"] == "24"
    labels = [n.all_text() for n in form.walk() if n.tag == "label"]
    assert labels == ["GPU", "Manual VRAM (GB)", "Context"]
    button = _one(form, "button")
    assert button.all_text() == "Recompute"
    assert "btn-primary" in button.classes() and button.attrs["type"] == "submit"
    assert "aria-invalid" not in fields["vram_gb"].attrs


def test_detail_input_error_is_one_alert_and_marks_the_field(client: TestClient) -> None:
    root = _tree(client.get(ERROR_DETAIL).text)
    alerts = [n for n in root.walk() if n.attrs.get("role") == "alert"]
    assert len(alerts) == 1
    assert "Invalid value for vram_gb." in alerts[0].all_text()
    vram = next(n for n in root.walk() if n.attrs.get("name") == "vram_gb")
    assert vram.attrs["aria-invalid"] == "true"


def test_detail_table_keeps_seven_columns_with_labels_and_explicit_roles(
    client: TestClient,
) -> None:
    root = _tree(client.get(GATED_DETAIL).text)
    table = _one(root, "table")
    assert table.attrs["role"] == "table"
    heads = [n for n in table.walk() if n.tag == "th"]
    assert [n.all_text() for n in heads] == VARIANT_HEADINGS
    assert all(n.attrs["role"] == "columnheader" for n in heads)
    rows = [n for n in table.walk() if n.tag == "tr"]
    assert len(rows) == 5 and all(n.attrs["role"] == "row" for n in rows)
    for row in rows[1:]:
        cells = [n for n in row.children if n.tag == "td"]
        assert [n.attrs["data-label"] for n in cells] == VARIANT_HEADINGS
        assert all(n.attrs["role"] == "cell" for n in cells)
        # Provenance is the last cell, after the figures and the verdict.
        assert cells[-1].attrs["data-label"] == "Provenance"
    labels = [rows[i].children[0].all_text() for i in range(1, 5)]
    assert labels == ["FP16 (FP16)", "Q4 (Q4) *", "Q5 (Q5)", "Q8 (Q8)"]
    assert "table-scroll" not in client.get(GATED_DETAIL).text


def test_detail_rows_show_required_available_and_margin(client: TestClient) -> None:
    root = _tree(client.get(OPEN_DETAIL).text)
    row = [n for n in root.walk() if n.tag == "tr"][2]
    values = [c.all_text() for c in row.children]
    assert len(values) == 7
    required, available, margin = (float(v) for v in values[2:5])
    assert available == 24.0
    assert round(available - required, 2) == margin
    assert values[5] == "Fits"


def test_detail_without_sources_omits_the_sources_block(
    client: TestClient, seeded_session: Session
) -> None:
    seeded_session.get(main.Model, "model_qwen25_7b_instruct").sources = None
    seeded_session.commit()
    root = _tree(client.get(OPEN_DETAIL).text)
    assert not _by_class(root, "sources")
    assert "KV cache estimate per 1k context" in root.all_text()
    assert len(_by_class(root, "facts")) == 1


def test_detail_css_follows_the_mockup() -> None:
    css = _css()
    assert ".table-scroll" not in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in _rules(css, ".facts")[0]
    assert "attr(data-label)" in css
    assert "overflow-wrap: anywhere" in _rules(css, ".sources-list li")[0]
    select = _rules(css, ".recompute select")
    assert select and all("ellipsis" not in rule for rule in select)
    raw = STYLES.read_text(encoding="utf-8").split("===== model detail (T-018)")[1]
    assert "text-overflow: ellipsis" not in raw
    assert "font-size: 16px" in _rules(css, ".recompute input[type=number], .recompute select")[0]


def test_asset_version_moved_on_for_the_detail_page() -> None:
    assert main.ASSET_VERSION > "20261025n"


# ---- T-021: the GPU suggestion list ----


def test_gpu_field_keeps_the_native_list_for_the_no_javascript_path(
    client: TestClient, seeded_session: Session
) -> None:
    root = _tree(client.get("/").text)
    search = _by_id(root, "gpu_search")
    assert search.attrs["list"] == "gpu-options"
    assert "combo" in search.parent.classes()
    # The script adds all of these; the server markup must not carry them.
    for attr in ("role", "aria-expanded", "aria-controls", "aria-activedescendant"):
        assert attr not in search.attrs, attr
    assert not [n for n in root.walk() if n.attrs.get("id") == "gpu-listbox"]
    options = [n for n in _by_id(root, "gpu-options").walk() if n.tag == "option"]
    gpus = main._load_gpus(seeded_session)
    assert len(options) == len(gpus) == 35
    assert [o.attrs["data-gpu-id"] for o in options] == [g.id for g in gpus]
    assert [o.attrs["value"] for o in options] == [
        f"{g.vendor} {g.name} ({g.vram_gb:.1f} GB)" for g in gpus
    ]


def test_app_js_gpu_combobox_follows_adr_0005() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    body = js.split("function initGpuCombobox")[1].split("function initExampleChips")[0]
    # Takes over from the native list only when it can: it removes `list` and sets the ARIA state.
    assert 'input.removeAttribute("list")' in body
    assert 'input.setAttribute("list", nativeList)' in body
    for fragment in (
        '"role", "combobox"',
        '"aria-autocomplete", "list"',
        '"aria-expanded"',
        '"aria-controls", "gpu-listbox"',
        '"aria-activedescendant"',
        'listbox.id = "gpu-listbox"',
        '"aria-label", "GPU suggestions"',
        '"role", "option"',
        '"aria-selected", "true"',
    ):
        assert fragment in body, fragment
    # Keys and pointer.
    for key in ("ArrowDown", "ArrowUp", "Enter", "Escape"):
        assert f'event.key === "{key}"' in body, key
    assert re.search(r"event\.preventDefault\(\);\s*choose\(active\);", body)
    assert 'addEventListener("pointerdown"' in body
    assert 'addEventListener("blur", close)' in body
    # Choosing reuses the exact-match code through input and change; it never submits.
    assert 'new Event("input"' in body and 'new Event("change"' in body
    assert "submit" not in body
    assert "initGpuCombobox();" in js.split('addEventListener("DOMContentLoaded"')[1]


def test_gpu_suggestion_css_follows_the_mockup() -> None:
    css = _css()
    box = _rules(css, ".listbox")[0]
    assert "position: absolute" in box and "left: 0" in box and "right: 0" in box
    assert "max-height: 300px" in box
    assert _rules(css, ".f-gpu")[0].strip() == "position: relative;"
    option = _rules(css, ".listbox li")[0]
    assert "min-height: 48px" in option and "overflow-wrap: anywhere" in option
    assert "text-overflow" not in option and "white-space: nowrap" not in option
    solid = re.search(r"@media \(max-width: 899px\)\s*\{\s*\.listbox\s*\{([^}]*)\}", css)
    assert solid and "var(--panel-solid)" in solid.group(1)
    assert "backdrop-filter: none" in solid.group(1)
    assert 'li[aria-selected="true"]' in css


def test_home_vram_field_is_marked_only_when_the_server_rejects_it(client: TestClient) -> None:
    bad = _tree(client.get("/results?vram_gb=abc").text)
    assert _by_id(bad, "vram_gb").attrs["aria-invalid"] == "true"
    other = _tree(client.get("/results?vram_gb=8&min_params_b=x").text)
    assert "aria-invalid" not in _by_id(other, "vram_gb").attrs
    assert "aria-invalid" not in _by_id(_tree(client.get("/").text), "vram_gb").attrs
    rule = _rules(_css(), '.form-pane input[aria-invalid="true"]')
    assert rule and "border-color: var(--wont-fg)" in rule[0] and "border-width: 2px" in rule[0]


def test_summary_partial_marks_whether_the_vram_field_is_invalid(client: TestClient) -> None:
    # An in-place refresh never re-renders the form, so the summary carries the marker app.js reads.
    def marker(query: str) -> str:
        for path in ("/results/partials/summary", "/results/partials/content"):
            tree = _tree(client.get(f"{path}?{query}", headers={"HX-Request": "true"}).text)
            banner = _by_id(tree, "summary-banner")
            assert banner is not None, path
            yield banner.attrs["data-vram-invalid"]

    assert set(marker("vram_gb=abc")) == {"true"}
    assert set(marker("vram_gb=0")) == {"true"}
    assert set(marker("vram_gb=24")) == {"false"}
    assert set(marker("vram_gb=8&min_params_b=x")) == {"false"}
    assert set(marker("")) == {"false"}
    js = APP_JS.read_text(encoding="utf-8")
    assert re.search(r'addEventListener\(\s*"htmx:afterSettle",\s*syncVramInvalid', js)
    body = js.split("function syncVramInvalid")[1].split("function initVramInvalid")[0]
    assert "dataset.vramInvalid" in body
    assert 'setAttribute("aria-invalid", "true")' in body
    assert 'removeAttribute("aria-invalid")' in body
    assert "initVramInvalid();" in js.split('addEventListener("DOMContentLoaded"')[1]


def test_choices_and_presets_carry_the_label_letter_spacing() -> None:
    css = _css()
    assert "letter-spacing: .01em" in _rules(css, ".choice")[0]
    assert "letter-spacing: .01em" in _rules(css, ".seg label")[0]


def test_asset_version_moved_on_for_the_suggestion_list() -> None:
    assert main.ASSET_VERSION > "20261027a"
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)
