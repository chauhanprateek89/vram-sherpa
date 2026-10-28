"""R-083: `/` renders landing.html, the box, chips and unseeded notice, with no results."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def unseeded_client(app: FastAPI, db_session: Session) -> TestClient:
    return TestClient(app)


def _main(html: str) -> str:
    return re.search(r"<main.*?</main>", html, re.S).group(0)


def _options(html: str) -> list[str]:
    return re.findall(r"<option [^>]*></option>", html)


def test_landing_structure(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    page_html = response.text
    body = _main(page_html)
    assert "<title>VRAM Sherpa</title>" in page_html
    assert '<main class="landing">' in page_html
    assert body.count("<h1") == 1
    assert "Which local LLMs fit your GPU?" in body
    assert "Type a GPU name, or enter its VRAM, like 12 GB." in body
    assert '<form id="landing-form" method="get" action="/results">' in body
    assert "hx-" not in body
    assert 'role="combobox"' not in body
    assert body.count("<input") == 1
    assert body.count('type="submit"') == 1
    assert "Check compatibility" in body
    assert re.search(r'<label for="gpu_search" class="sr-only">Search GPU</label>', body)
    for attr in ('list="gpu-options"', "autofocus", "data-submit-on-choose", 'autocomplete="off"'):
        assert attr in body
    assert 'placeholder="Your GPU, or its VRAM"' in body
    assert 'aria-current="page"' in page_html


def test_landing_has_no_results_chrome(client):
    body = _main(client.get("/").text)
    for absent in (
        "summary-banner",
        "results-shell",
        "details",
        "Apply filters",
        'id="vram_gb"',
        'id="gpu_id"',
        'context_tokens"',
        "clear-gpu",
        "chip-example",
        'role="alert"',
        "No GPU or VRAM selected yet.",
        "Top picks",
    ):
        assert absent not in body, absent


def test_options_match_results_page(client):
    landing = _options(client.get("/").text)
    results = _options(client.get("/results?vram_gb=8").text)
    assert len(landing) == 35
    assert landing == results


def test_chips_and_targets(client):
    body = _main(client.get("/").text)
    found = re.findall(r'<a class="chip" href="([^"]+)">([^<]+)</a>', body)
    hrefs = [(html.unescape(href), label) for href, label in found]
    assert [label for _, label in hrefs] == ["RTX 4060 8GB", "8GB", "24GB"]
    assert hrefs[1][0] == "/results?vram_gb=8&context_tokens=2048"
    assert hrefs[2][0] == "/results?vram_gb=24&context_tokens=2048"
    assert re.fullmatch(r"/results\?gpu_id=[^&]+&context_tokens=2048", hrefs[0][0])
    expected = [
        "NVIDIA GeForce RTX 4060 (8.0 GB)",
        "Manual VRAM (8 GB)",
        "Manual VRAM (24 GB)",
    ]
    for (href, _), hardware in zip(hrefs, expected, strict=True):
        page = client.get(href)
        assert page.status_code == 200
        assert hardware in page.text
        assert "<span data-fcount>0</span>" in page.text


@pytest.mark.parametrize("path", ["/?vram_gb=24&context_tokens=4096", "/?context_tokens=abc"])
def test_query_params_ignored(client, path):
    response = client.get(path, follow_redirects=False)
    assert response.status_code == 200
    assert "location" not in response.headers
    assert _main(response.text) == _main(client.get("/").text)


def test_home_does_not_build_results(client, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("home must not build results")

    monkeypatch.setattr(main, "_build_results", boom)
    monkeypatch.setattr(main, "_resolve_available_vram", boom)
    assert client.get("/").status_code == 200


def test_unseeded_landing(unseeded_client):
    response = unseeded_client.get("/")
    assert response.status_code == 200
    body = _main(response.text)
    assert '<div class="notice" role="status">' in body
    assert "Catalog is not initialized yet." in body
    assert "Seed the catalog data, then refresh this page." in body
    assert '<span class="chip" aria-disabled="true">Example GPU</span>' in body
    assert body.index("launch-chips") < body.index("launch-note")
    hrefs = [html.unescape(h) for h in re.findall(r'<a class="chip" href="([^"]+)"', body)]
    assert hrefs == [
        "/results?vram_gb=8&context_tokens=2048",
        "/results?vram_gb=24&context_tokens=2048",
    ]


def test_stylesheet_defines_the_landing_rules():
    css = (Path(main.__file__).parent / "static" / "css" / "styles.css").read_text()
    for selector in (".launch", ".ask", ".go-label", "a.chip", "a.btn-clear"):
        assert re.search(rf"(^|[\s,}}]){re.escape(selector)}[\s,{{:.\[]", css, re.M), selector
    assert '.chip[aria-disabled="true"]' in css
    assert css.index("landing (T-029)") < css.index("Utilities and motion")


def test_landing_chips_get_the_base_chip_look():
    css = (Path(main.__file__).parent / "static" / "css" / "styles.css").read_text()
    rule = re.search(r"^([^{}\n]*\.launch-chips \.chip[^{}\n]*)\{([^}]*)\}", css, re.M)
    assert rule, ".launch-chips .chip has no base rule"
    assert "min-height: var(--tap)" in rule.group(2)


def test_app_js_submits_on_choose_and_guards_blank_landing_form() -> None:
    js = (Path(main.__file__).parent / "static" / "js" / "app.js").read_text()
    assert "data-submit-on-choose" in js
    assert "requestSubmit" in js
    assert "landing-form" in js


def test_asset_version_bumped_for_landing_script() -> None:
    assert main.ASSET_VERSION > "20261028c"
    assert main.ASSET_VERSION != "20261027e"


def test_clear_on_results_is_a_link_to_an_empty_landing(client):
    html_ = client.get("/results?vram_gb=24").text
    assert '<a id="clear-gpu" class="btn-clear" href="/">Clear</a>' in html_
    page = client.get("/")
    assert page.status_code == 200
    assert '<form id="landing-form"' in page.text
    field = re.search(r"<input[^>]*id=\"gpu_search\"[^>]*>", page.text, re.S).group(0)
    assert not re.search(r'value="[^"]+"', field)


def test_app_js_has_no_clear_handler():
    js = (Path(main.__file__).parent / "static" / "js" / "app.js").read_text()
    assert "clear-gpu" not in js


def _landing_css() -> str:
    css = (Path(main.__file__).parent / "static" / "css" / "styles.css").read_text()
    start = css.index("landing (T-029)")
    return css[start : css.index("Utilities and motion", start)]


def _rule(css: str, selector: str) -> str:
    found = re.search(rf"^{re.escape(selector)} \{{([^}}]*)\}}", css, re.M)
    assert found, selector
    return found.group(1)


def test_landing_box_has_the_inset_field_look():
    """T-033: the box fills its form, so the submit sits inside its right edge, and wears
    the mockup's field look; the generic field rules are scoped to the results form."""
    box = _rule(_landing_css(), ".ask input[type=text]")
    for decl in (
        "width: 100%",
        "border: 1.5px solid var(--line)",
        "background: var(--field)",
        "box-shadow: var(--sh-in)",
        "border-radius: 22px",
        "padding: 0 70px 0 50px",
    ):
        assert decl in box, decl
    assert "color: var(--muted)" in _rule(_landing_css(), ".ask input::placeholder")


def test_landing_greeting_has_no_default_margins():
    """T-033: the mockup's reset zeroes h1 and p margins; the app's base does not, so the
    landing pane resets its own to keep the h1-to-help-line gap and the pane height."""
    assert "margin: 0" in _rule(_landing_css(), ".launch h1, .launch p")


def test_landing_rules_stay_scoped_to_the_landing_page():
    """T-033: nothing in the landing section restyles the /results form or its combobox."""
    css = re.sub(r"/\*.*?\*/", "", _landing_css(), flags=re.S)
    for selector in re.findall(r"([^{}@]+)\{", css):
        for part in selector.split(","):
            part = part.strip()
            if not part or part.startswith("(") or part[0].isdigit():
                continue
            assert not re.match(r"(input|select|form|\.combo|\.form-pane|\.field-row)\b", part), (
                part
            )


def test_asset_version_bumped_for_landing_fix() -> None:
    assert main.ASSET_VERSION > "20261028e"
