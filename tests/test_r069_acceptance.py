"""QA acceptance checks for R-069 (the answer leads the results). Black-box, over HTTP.

Pixel positions are measured in a browser (see .sdlc/qa/R-069/report.md); these check what the
markup can prove: order, closed filters without JavaScript, the filter count, and the copy.
"""

from __future__ import annotations

import re
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

REF = "/results?vram_gb=24&context_tokens=4096"
STATES = [
    "/results?gpu_id=no_such_gpu",
    REF,
    "/results?gpu_search=RTX",
    "/results?gpu_search=" + quote("RTX 3060 12GB"),
    "/results?vram_gb=abc",
    REF + "&family=qwen&quant_bucket=Q4",
]
OLD_PICK_COPY = [
    "Highest-ranked variant that remains in the Fits band.",
    "Next highest-ranked practical option by current ranking order.",
    "Lowest-ranked viable option, usually the lightest footprint.",
]
NEW_PICK_COPY = [
    "The strongest model that still leaves room to spare.",
    "Nearly as capable, and comfortable to run.",
    "The lightest model that still runs on your GPU.",
]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("url", STATES)
def test_headline_precedes_chips_picks_and_cards(client: TestClient, url: str) -> None:
    html = client.get(url + "&verdict=fits").text
    headline = html.index('id="ans-h"')
    picks = html.index('class="picks')
    first_card = html.index('<article class="model-card')
    assert headline < picks < first_card
    if "Active filters:" in html:
        assert headline < html.index("Active filters:") < picks


@pytest.mark.parametrize("url", STATES)
def test_filters_render_closed_in_the_html(client: TestClient, url: str) -> None:
    html = client.get(url).text
    tags = re.findall(r"<details[^>]*class=\"filters[^>]*>", html)
    assert len(tags) == 1
    assert " open" not in tags[0]


@pytest.mark.parametrize(
    ("query", "count"),
    [
        ("vram_gb=24&context_tokens=4096", 0),
        ("vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4", 2),
        (
            "vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4"
            "&min_params_b=8&max_params_b=8",
            4,
        ),
    ],
)
def test_filters_label_counts_the_chips(client: TestClient, query: str, count: int) -> None:
    html = client.get("/results?" + query).text
    assert f"Filters (<span data-fcount>{count}</span>)" in html
    assert html.count('class="chip-link"') == count


def test_filter_controls_are_in_the_page_behind_the_disclosure(client: TestClient) -> None:
    html = client.get(REF).text
    block = html.split('<details class="filters', 1)[1].split("</details>", 1)[0]
    for name in ("family", "quant_bucket", "min_params_b", "max_params_b"):
        assert f'name="{name}"' in block


@pytest.mark.parametrize("url", STATES)
def test_top_pick_descriptions_are_the_approved_copy(client: TestClient, url: str) -> None:
    html = client.get(url).text
    descs = re.findall(r'<p class="pick-desc">([^<]*)</p>', html)
    assert descs == NEW_PICK_COPY
    for text in descs:
        assert "ranked" not in text and "ranking" not in text
    for old in OLD_PICK_COPY:
        assert old not in html


@pytest.mark.parametrize(
    ("query", "line"),
    [
        ("vram_gb=1&context_tokens=4096&quant_bucket=Q8", "There are 95 models in Won't fit"),
        ("vram_gb=1&context_tokens=4096&family=yi", "There is 1 model in Won't fit"),
    ],
)
def test_empty_state_counts_models_and_keeps_the_singular(
    client: TestClient, query: str, line: str
) -> None:
    html = client.get("/results?" + query).text
    assert line + ", hidden by default." in html
    assert "variants" not in html.split('id="results-list-shell"', 1)[1]


def _children(html: str, open_tag: str) -> list[str]:
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.depth = 0
            self.target = -1
            self.out: list[str] = []

        def handle_starttag(self, tag, attrs):
            if self.target < 0 and open_tag in self.get_starttag_text():
                self.target = self.depth
            elif self.depth == self.target + 1 and self.target >= 0:
                cls = dict(attrs).get("class", "").split()
                self.out.append(tag + ("." + cls[0] if cls and tag != "article" else ""))
            if tag not in ("br", "img", "input", "meta", "link", "use", "path"):
                self.depth += 1

        def handle_startendtag(self, tag, attrs):
            pass

        def handle_endtag(self, tag):
            self.depth -= 1
            if self.target >= 0 and self.depth == self.target:
                self.target = 10**9

    p = P()
    p.feed(html)
    return p.out


def test_answer_tile_children_and_document_order(client: TestClient) -> None:
    html = client.get(REF + "&verdict=fits").text
    assert _children(html, 'id="summary-banner"') == [
        "h2.headline",
        "p.ctx-line",
        "div.verdicts",
        "p.note",
    ]
    order = [
        html.index('id="ans-h"'),
        html.index('class="verdicts'),
        html.index('class="note'),
        html.index('class="picks'),
        html.index('id="list-heading"'),
        html.index('class="sizes"'),
        html.index("data-model-card"),
    ]
    assert order == sorted(order)


def test_list_pane_children(client: TestClient) -> None:
    html = client.get(REF + "&verdict=fits").text
    kids = _children(html, 'id="results-cards"')
    assert kids[:2] == ["div.list-head", "div.card-grid"]
    assert set(kids[2:]) <= {"nav.pager"}


def test_list_css_values_are_pinned() -> None:
    from pathlib import Path

    css = (Path(__file__).parents[1] / "src/vramsherpa/static/css/styles.css").read_text()
    for rule in (
        ".list-pane { padding: 14px 10px 12px; }",
        "@media (min-width: 900px) { .list-pane { padding: 10px 14px 14px; } }",
        "padding: 0 6px 8px; }",
        "font-size: 24px; line-height: 1.15;",
        "min-height: 44px; min-width: 52px;",
        ".list-pane .card-grid { padding: 0; }",
        "#list-heading { scroll-margin-top: 96px; }",
        ".list-head h2:focus { outline: none; }",
        ".list-head h2:focus-visible { outline: 3px solid var(--focus);",
    ):
        assert rule in css, rule
    assert css.index(".results { padding: 10px;") < css.index(".list-pane { padding")
    assert css.index("@media (min-width: 900px) { .results { padding: 12px; } }") < css.index(
        "@media (min-width: 900px) { .list-pane"
    )
