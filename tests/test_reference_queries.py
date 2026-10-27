"""Golden test for R-078: the reference queries keep their models, order and figures.

The expected values in ``fixtures/r078_reference.json`` were captured once from the tree
whose ``src/`` equals ``main`` at f2ad2d8, before the overhaul. The test never rewrites
the file.

The extractor reads only hooks the overhaul keeps: ``data-model-card``,
``data-variant-pane``, the ``/models/<id>`` link, the gauge's ``role="img"``
``aria-label`` (R-018, kept by R-072) and the protected text (``Fits``, ``Tight``,
``Won't fit``, ``<n> GB spare``, ``<n> GB over``, ``<n> GB needed``), with tags
stripped. It must keep working, unchanged, after every template change.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette.requests import Request

from vramsherpa.main import _catalog_version, _results_page_context

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "r078_reference.json"

REFERENCE_QUERIES = [
    "/results?vram_gb=24&context_tokens=4096",
    "/results?vram_gb=12&context_tokens=8192&family=qwen&quant_bucket=Q4",
]

_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}
_VERDICT_RE = re.compile(r"(Won't fit|Fits|Tight)")
_MARGIN_RE = re.compile(r"(\d+(?:\.\d+)?) GB (spare|over)")
_NEEDED_RE = re.compile(r"(\d+(?:\.\d+)?) GB needed")
_GAUGE_RE = re.compile(
    r"Needs (\d+(?:\.\d+)?) of (\d+(?:\.\d+)?) gigabytes, "
    r"with (\d+(?:\.\d+)?) gigabytes held in reserve"
)
_MODEL_HREF_RE = re.compile(r"^/models/([^/?#]+)")


class _CardExtractor(HTMLParser):
    """Collect, per ``data-model-card``, the model link and each variant pane's text and gauge."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict] = []
        self._stack: list[str] = []
        self._card_depth: int | None = None
        self._pane_depth: int | None = None
        self._link_open = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag not in _VOID_TAGS:
            self._stack.append(tag)
        depth = len(self._stack)

        if "data-model-card" in attributes and self._card_depth is None:
            self._card_depth = depth
            self.cards.append({"model_id": None, "name": "", "panes": []})
            return
        if self._card_depth is None:
            return

        card = self.cards[-1]
        if tag == "a" and card["model_id"] is None:
            match = _MODEL_HREF_RE.match(attributes.get("href") or "")
            if match:
                card["model_id"] = match.group(1)
                self._link_open = True
        if "data-variant-pane" in attributes and self._pane_depth is None:
            self._pane_depth = depth
            card["panes"].append(
                {"variant_id": attributes["data-variant-pane"], "text": [], "gauge": None}
            )
        if self._pane_depth is not None and attributes.get("role") == "img":
            pane = card["panes"][-1]
            if pane["gauge"] is None:
                pane["gauge"] = attributes.get("aria-label")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or tag not in self._stack:
            return
        while self._stack:
            popped = self._stack.pop()
            depth = len(self._stack) + 1
            if popped == "a":
                self._link_open = False
            if self._pane_depth == depth:
                self._pane_depth = None
            if self._card_depth == depth:
                self._card_depth = None
            if popped == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._card_depth is None:
            return
        card = self.cards[-1]
        if self._link_open:
            card["name"] += data
        if self._pane_depth is not None:
            card["panes"][-1]["text"].append(data)


def extract_cards(html: str) -> list[dict]:
    """Return the cards of a results page as plain dicts: id, name, and one entry per pane."""
    parser = _CardExtractor()
    parser.feed(html)
    parser.close()

    cards = []
    for raw_card in parser.cards:
        panes = []
        for raw_pane in raw_card["panes"]:
            text = " ".join(" ".join(raw_pane["text"]).split())
            verdict = _VERDICT_RE.search(text)
            margin = _MARGIN_RE.search(text)
            needed = _NEEDED_RE.search(text)
            gauge = _GAUGE_RE.fullmatch(raw_pane["gauge"] or "")
            assert verdict, f"no verdict in pane {raw_pane['variant_id']}: {text!r}"
            assert margin, f"no margin in pane {raw_pane['variant_id']}: {text!r}"
            assert needed, f"no required VRAM in pane {raw_pane['variant_id']}: {text!r}"
            assert gauge, f"no gauge label in pane {raw_pane['variant_id']}: {raw_pane['gauge']!r}"
            panes.append(
                {
                    "variant_id": raw_pane["variant_id"],
                    "verdict": verdict.group(1),
                    "required": f"{needed.group(1)} GB needed",
                    "available_gb": gauge.group(2),
                    "margin": f"{margin.group(1)} GB {margin.group(2)}",
                    "gauge": raw_pane["gauge"],
                }
            )
        cards.append(
            {
                "model_id": raw_card["model_id"],
                "name": " ".join(raw_card["name"].split()),
                "panes": panes,
            }
        )
    return cards


def _request(app, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "app": app,
            "router": app.router,
        }
    )


def computed_cards(app, session: Session, url: str) -> list[dict]:
    """The cards the app computes for ``url``, in the extractor's shape."""
    parts = urlsplit(url)
    query = parse_qs(parts.query)

    def first(name: str) -> str | None:
        return query[name][0] if name in query else None

    context = _results_page_context(
        _request(app, parts.path),
        session,
        gpu_id=first("gpu_id"),
        gpu_search=first("gpu_search"),
        vram_gb=first("vram_gb"),
        context_tokens=int(first("context_tokens") or 2048),
        family=query.get("family", []),
        quant_bucket=query.get("quant_bucket", []),
        min_params_b=first("min_params_b"),
        max_params_b=first("max_params_b"),
        recommended_only=first("recommended_only") in {"1", "true", "True", "on"},
    )
    cards = []
    for card in context["model_cards"]:
        panes = []
        for variant in card.variants:
            margin = variant.margin_gb
            panes.append(
                {
                    "variant_id": variant.variant_id,
                    "verdict": variant.classification,
                    "required": f"{variant.required_vram_gb:.2f} GB needed",
                    "available_gb": f"{variant.available_vram_gb:.1f}",
                    "margin": (
                        f"{margin:.1f} GB spare" if margin >= 0 else f"{-margin:.1f} GB over"
                    ),
                    "gauge": (
                        f"Needs {variant.required_vram_gb:.2f} of "
                        f"{variant.available_vram_gb:.1f} gigabytes, with "
                        f"{variant.reserve_gb:.2f} gigabytes held in reserve"
                    ),
                }
            )
        cards.append({"model_id": card.model_id, "name": card.model_name, "panes": panes})
    return cards


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_covers_both_reference_queries() -> None:
    fixture = _load_fixture()

    assert sorted(fixture["queries"]) == sorted(REFERENCE_QUERIES)
    for cards in fixture["queries"].values():
        assert cards, "a reference query with no cards would make the golden check vacuous"


@pytest.mark.parametrize("url", REFERENCE_QUERIES)
def test_rendered_cards_match_the_f2ad2d8_fixture(
    client: TestClient, seeded_session: Session, url: str
) -> None:
    fixture = _load_fixture()
    seeded_version = _catalog_version(seeded_session)
    if seeded_version != fixture["catalog_version"]:
        pytest.skip(
            f"{FIXTURE_PATH.name} was captured with catalog_version "
            f"{fixture['catalog_version']}, but the seeded catalog is {seeded_version}; "
            "the computed comparison still runs"
        )

    response = client.get(url)

    assert response.status_code == 200
    assert extract_cards(response.text) == fixture["queries"][url]


@pytest.mark.parametrize("url", REFERENCE_QUERIES)
def test_rendered_cards_match_the_computed_model_cards(
    app, client: TestClient, seeded_session: Session, url: str
) -> None:
    response = client.get(url)

    assert response.status_code == 200
    rendered = extract_cards(response.text)
    assert rendered, "the reference query rendered no cards"
    assert rendered == computed_cards(app, seeded_session, url)
