"""QA acceptance checks for R-076 (header and footer). Black-box, over HTTP, plus the stylesheet.

Phone stacking, the render against the approved mockups and pixel overlap are browser checks; see
.sdlc/qa/R-076/report.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

ROOT = Path(__file__).resolve().parents[1]
STYLES = (ROOT / "src" / "vramsherpa" / "static" / "css" / "styles.css").read_text(encoding="utf-8")
PAGES = {
    "/": "Estimator",
    "/results?vram_gb=24&context_tokens=4096": "Estimator",
    "/results?vram_gb=abc": "Estimator",
    "/models/model_llama_31_8b_instruct?vram_gb=24": "Estimator",
    "/models/model_qwen25_7b_instruct?vram_gb=abc": "Estimator",
    "/how-it-works": "How it works",
}


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _region(page: str, tag: str) -> str:
    match = re.search(rf"<{tag}\b.*?</{tag}>", page, re.S)
    assert match, tag
    return match.group(0)


def _normalised(fragment: str) -> str:
    fragment = re.sub(r'\s*aria-current="page"', "", fragment)
    fragment = re.sub(r"\bis-active\b", "", fragment)
    return " ".join(fragment.split())


@pytest.mark.parametrize("url", PAGES)
def test_logo_mark_and_wordmark_are_one_link_to_home(client: TestClient, url: str) -> None:
    header = _region(client.get(url).text, "header")
    brand = re.search(r'<a class="brand" href="/">(.*?)</a>', header, re.S)
    assert brand, "brand link to / missing"
    inner = brand.group(1)
    assert '<svg class="mark"' in inner and 'aria-hidden="true"' in inner
    assert 'focusable="false"' in inner
    assert '<span class="brand-name">VRAM Sherpa</span>' in inner


@pytest.mark.parametrize(("url", "current"), list(PAGES.items()))
def test_navigation_marks_exactly_the_right_link(
    client: TestClient, url: str, current: str
) -> None:
    nav = _region(client.get(url).text, "nav")
    marked = re.findall(r'<a\b[^>]*aria-current="page"[^>]*>\s*(.*?)\s*</a>', nav, re.S)
    assert marked == [current]
    assert re.findall(r'href="([^"]+)"', nav) == ["/", "/how-it-works"]


def test_header_and_footer_markup_is_identical_across_pages(client: TestClient) -> None:
    shells = set()
    for url in PAGES:
        page = client.get(url).text
        shells.add((_normalised(_region(page, "header")), _normalised(_region(page, "footer"))))
    assert len(shells) == 1


def test_theme_toggle_is_a_pressed_button_and_the_theme_script_runs_before_the_stylesheet(
    client: TestClient,
) -> None:
    page = client.get("/").text
    toggle = re.search(r'<button id="theme-toggle"[^>]*>(.*?)</button>', page, re.S)
    assert toggle and 'aria-pressed="false"' in toggle.group(0)
    assert toggle.group(1).strip() == "Dark mode"
    assert page.index("vramsherpa-theme") < page.index('rel="stylesheet"')


@pytest.mark.parametrize("url", PAGES)
def test_footer_carries_the_three_values(client: TestClient, url: str) -> None:
    footer = _region(client.get(url).text, "footer")
    assert re.findall(r"<span>(.*?)</span>", footer) == [
        "Catalog version: 2026-10-23",
        f"Estimation policy: {main.ESTIMATION_POLICY_VERSION}",
        "Last updated: 2026-10-23",
    ]


def test_footer_says_unseeded_on_an_empty_catalog(app: FastAPI, db_session: Session) -> None:
    client = TestClient(app)
    for url in ("/", "/how-it-works"):
        footer = _region(client.get(url).text, "footer")
        assert re.findall(r"<span>(.*?)</span>", footer) == [
            "Catalog version: unseeded",
            f"Estimation policy: {main.ESTIMATION_POLICY_VERSION}",
            "Last updated: unseeded",
        ]


def test_footer_stacks_its_values_on_a_phone_and_never_uses_nowrap() -> None:
    phone = re.search(r"@media \(max-width: 560px\)\s*\{\s*\.footer\s*\{([^}]*)\}", STYLES)
    assert phone, "no phone rule for .footer"
    assert "flex-direction: column" in phone.group(1)
    assert "width: 100%" in phone.group(1)
    footer_rule = re.search(r"^\.footer\s*\{([^}]*)\}", STYLES, re.M)
    assert footer_rule and "flex-wrap: wrap" in footer_rule.group(1)
    assert "nowrap" not in footer_rule.group(1) + phone.group(1)
