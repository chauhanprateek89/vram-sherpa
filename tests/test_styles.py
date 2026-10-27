"""T-008: design tokens, the self-hosted font, Google Fonts removed (ADR-0001, ADR-0002).

T-020: one 3 px focus ring, reduced motion and transparency, no legacy rules, ADR-0001's section
order and the font-size and nowrap guards (R-070, R-077).
"""

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
TEMPLATES_DIR = ROOT / "src" / "vramsherpa" / "templates"
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


# ---- T-020 ----


def _flat_rules(css: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """Every style rule as (selector list, declarations, enclosing at-rules), at-rules opened."""
    found: list[tuple[str, str, tuple[str, ...]]] = []

    def walk(text: str, context: tuple[str, ...]) -> None:
        position = 0
        while match := re.compile(r"([^{}]+)\{").search(text, position):
            selector = " ".join(match.group(1).split())
            body = _block_body(text, match.end() - 1)
            position = match.end() + len(body) + 1
            if selector.startswith(("@media", "@supports")):
                walk(body, (*context, selector))
            elif not selector.startswith("@"):
                found.append((selector, body, context))

    walk(css, ())
    return found


def _selectors(selector_list: str) -> list[str]:
    return [part.strip() for part in selector_list.split(",")]


# The short figure and tag selectors docs/design/build/components.css keeps on one line (R-070).
NOWRAP_ALLOWED = {
    ".sr-only",
    ".tag-gated",
    ".why-list dd",
    ".vtable td.fig",
    ".vtable td.vl",
    ".kv-list dd",
    ".steps .out",
}


def test_nowrap_only_on_short_figures_and_tags() -> None:
    nowrap = {
        selector
        for selector_list, body, _ in _flat_rules(_css())
        if re.search(r"white-space:\s*nowrap", body)
        for selector in _selectors(selector_list)
    }
    assert nowrap, "the allowlisted figures and tags keep white-space: nowrap"
    assert sorted(nowrap - NOWRAP_ALLOWED) == []
    # Names, labels, URLs and formulas wrap: none of their rules says nowrap.
    for selector_list, body, _ in _flat_rules(_css()):
        if any(word in selector_list for word in ("h1", "h3", "label", "sources", "code", ".calc")):
            assert "nowrap" not in body, selector_list


def _smallest_token_px(css: str) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for name, value in re.findall(r"(--[\w-]+)\s*:\s*([\d.]+)px\s*;", css):
        sizes[name] = min(float(value), sizes.get(name, float("inf")))
    return sizes


def test_no_font_size_below_12px() -> None:
    css = _css()
    tokens = _smallest_token_px(css)
    smallest_base = min(tokens["--fs-caption"], tokens["--fs-small"])
    sizes = re.findall(r"(?<![\w-])font-size\s*:\s*([^;}]+)", css)
    assert len(sizes) > 50
    for raw in sizes:
        value = raw.strip()
        if value in {"inherit", "initial", "unset"}:
            continue
        if token := re.fullmatch(r"var\((--[\w-]+)\)", value):
            px = tokens[token.group(1)]
        elif number := re.fullmatch(r"([\d.]+)px", value):
            px = float(number.group(1))
        elif number := re.fullmatch(r"([\d.]+)rem", value):
            px = float(number.group(1)) * 16
        elif number := re.fullmatch(r"([\d.]+)em", value):
            # an em size is relative to its parent; the smallest parent text is the caption size
            px = float(number.group(1)) * smallest_base
        else:
            raise AssertionError(f"font-size the test cannot resolve: {value}")
        assert px >= 12, value


def test_one_three_px_focus_ring_on_every_control() -> None:
    css = _css()
    rules = _flat_rules(css)
    base = [body for selector, body, context in rules if selector == ":focus-visible"]
    assert len(base) == 1 and not [c for s, _, c in rules if s == ":focus-visible" and c]
    assert "outline: 3px solid var(--focus)" in base[0]
    assert "outline-offset: 2px" in base[0]
    # Every other outline is the same ring (on a wrapper) or none (its wrapper takes the ring).
    for value in re.findall(r"(?<![\w-])outline\s*:\s*([^;}]+)", css):
        assert value.strip() in {"3px solid var(--focus)", "none"}, value
    for value in re.findall(r"outline-offset\s*:\s*([^;}]+)", css):
        assert value.strip() in {"1px", "2px"}, value
    assert "outline: 2px" not in css


def test_reduced_motion_stops_every_transition_and_animation() -> None:
    reduced = [
        body
        for selector, body, context in _flat_rules(_css())
        if context == ("@media (prefers-reduced-motion: reduce)",)
        and _selectors(selector) == ["*", "*::before", "*::after"]
    ]
    assert len(reduced) == 1
    assert "transition: none !important" in reduced[0]
    assert "animation: none !important" in reduced[0]


def test_panes_turn_solid_without_blur_or_with_reduced_transparency() -> None:
    css = _css()
    rules = _flat_rules(css)
    no_blur = [
        body
        for selector, body, context in rules
        if context and context[0].startswith("@supports not ((backdrop-filter")
    ]
    assert no_blur == [" background: var(--panel-solid); "]
    reduced = [
        (selector, body)
        for selector, body, context in rules
        if context == ("@media (prefers-reduced-transparency: reduce)",)
    ]
    assert (".pane, .lite", reduced[0][1]) == reduced[0]
    assert "background: var(--panel-solid) !important" in reduced[0][1]
    assert "backdrop-filter: none !important" in reduced[0][1]
    assert reduced[1] == (".pane::before", " display: none; ")
    # The @supports fallback has no !important, so it has to come after every .lite rule.
    assert css.index("@supports not ((backdrop-filter") > max(
        match.start() for match in re.finditer(r"(?:\n|\{ )\.lite \{", css)
    )


LEGACY_ALIASES = (
    "--bg-strong",
    "--panel-alt",
    "--text",
    "--line-strong",
    "--brand",
    "--brand-strong",
    "--badge-",
    "--fit-bar",
    "--wont-fit-bar",
    "--reserve-ink",
    "--sans",
    "--mono",
)


def test_legacy_aliases_and_hard_coded_colours_are_gone() -> None:
    css = _css()
    for alias in LEGACY_ALIASES:
        assert not re.search(re.escape(alias) + r"(?![\w-])" if alias[-1] != "-" else alias, css)
    for colour in ("#f9be2d", "#e05050"):
        assert colour not in css.lower()


def test_pre_overhaul_rules_are_gone() -> None:
    css = _css()
    raw = STYLES.read_text(encoding="utf-8")
    for dead in (
        ".chip-active",
        ".results-cards",
        ".result-card",
        ".badge",
        ".vram-gauge",
        ".numbers-grid",
        ".advanced-disclosure",
        ".panel {",
        "select[multiple]",
        "font-family: inherit",
        ".recompute h2.sr-only",
        "line-height: 1.45",
    ):
        assert dead not in css, dead
    # No global link hover: it recoloured the brand and the filter chips, which the mockups do not.
    assert "a:hover" not in [s for s, _, _ in _flat_rules(css)]
    assert "tag_gated" not in raw
    assert "gated_tag macro" in raw
    # The formula and the classification rule share one code block declaration.
    shared = [s for s, _, _ in _flat_rules(css) if ".rule code" in _selectors(s)]
    assert shared == [".term code.formula, .rule code"]


def _template_sources() -> str:
    pages = "".join(p.read_text(encoding="utf-8") for p in TEMPLATES_DIR.glob("*.html"))
    return pages + (STATIC_DIR / "js" / "app.js").read_text(encoding="utf-8")


def test_every_class_and_id_in_the_stylesheet_is_used() -> None:
    """Notes for T-020: a rule goes only once nothing renders its class."""
    css = re.sub(r'url\("[^"]*"\)', "", _css())
    selectors = " ".join(selector for selector, _, _ in _flat_rules(css))
    sources = _template_sources()
    # htmx adds htmx-request itself; the verdict and bar classes are built from a slug.
    composed = {"htmx-request"} | {
        f"{p}-{k}" for p in ("state", "bar") for k in ("fits", "tight", "wont")
    }
    assert "state-{{" in sources and "bar-{{" in sources
    for name in set(re.findall(r"\.([a-zA-Z][\w-]*)", selectors)) - composed:
        assert re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", sources), name
    for name in set(re.findall(r"#([a-zA-Z][\w-]*)", selectors)):
        assert f'id="{name}"' in sources, name


SECTIONS = ("Tokens", "Reset and base", "Layout", "Components", "Page-specific rules", "Utilities")


def test_sections_follow_adr_0001_order() -> None:
    raw = STYLES.read_text(encoding="utf-8")
    starts = []
    for title in SECTIONS:
        heading = re.search(r"(?m)^   " + re.escape(title) + r"\b.*\n   =+ \*/", raw)
        assert heading, title
        starts.append(heading.start())
    assert starts == sorted(starts)
    tokens_end = starts[1]
    assert raw.index("@font-face") < tokens_end
    assert max(m.start() for m in re.finditer(r"(?m)^\s*:root\b", raw)) < tokens_end
    assert raw.index("*, *::before, *::after { box-sizing") > tokens_end


def test_at_most_two_breakpoints_noted_beside_the_tokens() -> None:
    css = _css()
    widths = {
        int(value) + (1 if kind == "max" else 0)
        for kind, value in re.findall(r"\((min|max)-width:\s*(\d+)px\)", css)
    }
    # max-width: 560px reads as "up to 560"; normalise both kinds to the first wide pixel
    breakpoints = {width - 1 if width == 561 else width for width in widths}
    assert len(breakpoints) <= 2, sorted(breakpoints)
    header = STYLES.read_text(encoding="utf-8").split("*/", 1)[0]
    for width in breakpoints:
        assert f"{width}px" in header, width


def test_box_sizing_reset_reaches_pseudo_elements_and_line_height_is_the_token() -> None:
    css = _css()
    assert re.search(r"(?m)^\*, \*::before, \*::after \{ box-sizing: border-box; \}", css)
    assert css.count("box-sizing") == 1, "per-rule box-sizing patches are not needed any more"
    body = [b for s, b, c in _flat_rules(css) if s == "body" and not c]
    assert len(body) == 1
    assert "line-height: var(--lh-body)" in body[0]
    assert "--lh-body: 1.5" in css


def test_unseeded_info_icon_cuts_out_the_sheet_colour() -> None:
    status = [b for s, b, _ in _flat_rules(_css()) if s == ".empty .empty-status"]
    assert len(status) == 1
    assert "--pill-bg: var(--panel-solid)" in status[0]
    icon = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
    assert re.search(r'id="i-info".*var\(--pill-bg', icon)


def test_asset_version_moved_on_for_t020() -> None:
    assert main.ASSET_VERSION > "20261025r"
