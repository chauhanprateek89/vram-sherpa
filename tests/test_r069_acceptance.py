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
    html = client.get(url).text
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
