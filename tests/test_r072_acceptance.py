"""QA acceptance checks for R-072 (model cards, quantisation switcher, fit gauge). Black-box.

Pixel measures (single-row switcher at 390 px, non-colour verdict cues, reserve band) are taken in
a browser; see .sdlc/qa/R-072/report.md. These check what the markup and stylesheet can prove.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
STYLES = (ROOT / "src" / "vramsherpa" / "static" / "css" / "styles.css").read_text()
URL = "/results?vram_gb=8&context_tokens=2048&family=llama&min_params_b=8&max_params_b=8"


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _rule(selector: str) -> str:
    match = re.search(rf"(?m)^{re.escape(selector)}\s*\{{([^}}]*)\}}", STYLES)
    assert match, f"no rule for {selector}"
    return match.group(1)


def test_every_card_has_ordered_buttons_one_pressed_and_a_gauge_per_pane(
    client: TestClient,
) -> None:
    html = client.get(URL).text
    cards = html.split("data-model-card")[1:]
    assert cards
    for card in cards:
        labels = re.findall(r"data-quant-btn[^>]*>\s*(Q4|Q5|Q8|FP16)\s*</button>", card, re.S)
        assert labels == [x for x in ("Q4", "Q5", "Q8", "FP16") if x in labels]
        assert card.count('aria-pressed="true"') == 1
        assert card.count('role="img"') == card.count("data-variant-pane=")


def test_won_t_fit_buttons_carry_tooltip_and_strike_through_rule(client: TestClient) -> None:
    html = client.get(URL).text
    assert re.search(r'data-oversize="true" title="Does not fit in 8 GB"', html)
    assert "line-through" in _rule(".quants button[data-oversize]")


def test_phone_switcher_is_a_four_column_single_row_grid() -> None:
    rule = _rule(".quants")
    assert "display: grid" in rule
    assert "repeat(4, minmax(0, 1fr))" in rule
    assert not re.search(r"flex-wrap|grid-template-rows", rule)


def test_verdicts_use_three_distinct_shapes_plus_the_label(client: TestClient) -> None:
    html = client.get(URL).text
    assert re.search(r'<use href="#i-tight"/></svg>Tight', html)
    sprite = client.get("/").text
    shapes = {
        name: re.search(rf'<symbol id="i-{name}"[^>]*>(.*?)</symbol>', sprite, re.S).group(1)
        for name in ("fits", "tight", "wont")
    }
    outline = {name: re.match(r"\s*<(\w+)", body).group(1) for name, body in shapes.items()}
    assert outline == {"fits": "circle", "tight": "path", "wont": "rect"}
    assert len(set(shapes.values())) == 3


def test_reserve_band_is_a_dashed_outline_not_a_fill() -> None:
    rule = _rule(".resv")
    assert "dashed" in rule
    assert re.search(r"top:\s*-\d+px", rule) and re.search(r"bottom:\s*-\d+px", rule)
    assert "var(--resv)" in rule
