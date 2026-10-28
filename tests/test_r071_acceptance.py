"""QA acceptance checks for R-071 (input form on `/` and `/results`). Black-box, over HTTP.

Pointer, keyboard and layout behaviour of the styled GPU list is checked in a real browser (see
.sdlc/qa/R-071/report.md). These check what the markup can prove: messages shown once, the native
list kept for no-JavaScript, the copy table's wording, and the protected message text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

DATA = Path(__file__).resolve().parents[1] / "data"
FORM_PAGES = [
    "/results?vram_gb=24&context_tokens=4096",
    "/results?gpu_search=RTX",
    "/results?vram_gb=abc",
]
REMOVED_FORM_COPY = [
    "Choose either your exact GPU or enter available VRAM directly.",
    "I know my GPU",
    "I know my VRAM",
    "If your GPU is not listed, leave this blank and enter VRAM instead.",
    "Start typing a GPU name (or leave blank)",
]
KEPT_FORM_COPY = [
    "Find models that fit your VRAM",
    "Pick your GPU, or enter your VRAM if it isn't listed.",
    "Search GPU",
    "Available VRAM (GB)",
    "Context window",
    "Check compatibility",
    "Start typing a GPU name",
]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _visible_text(html: str) -> str:
    html = re.sub(r"<(script|style|template)\b.*?</\1>", "", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _catalog_labels() -> list[str]:
    items = json.loads((DATA / "seed_gpus.json").read_text())["items"]
    items.sort(key=lambda g: (g["vram_gb"], g["name"]))
    return [f"{g['vendor']} {g['name']} ({g['vram_gb']:.1f} GB)" for g in items]


def test_input_error_list_appears_once_in_one_alert(client: TestClient) -> None:
    html = client.get("/results?vram_gb=abc").text
    assert html.count('role="alert"') == 1
    assert html.count("Please fix the following input issues:") == 1
    assert html.count("Invalid value for vram_gb.") == 1


def test_every_input_error_keeps_its_words_and_shows_once(client: TestClient) -> None:
    cases = {
        "/results?vram_gb=0": "vram_gb must be greater than or equal to 1e-06.",
        "/results?min_params_b=abc": "Invalid value for min_params_b.",
        "/results?max_params_b=-1": "max_params_b must be greater than or equal to 0.",
        "/results?min_params_b=10&max_params_b=5": (
            "min_params_b must be less than or equal to max_params_b."
        ),
    }
    for url, message in cases.items():
        html = client.get(url).text
        assert html.count('role="alert"') == 1, url
        assert html.count(message) == 1, (url, message)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("RTX", "matched multiple entries"),
        ("RTX 3060 12GB", "Using 12 GB inferred from your input."),
        ("Foo Bar", "Select a listed GPU or enter VRAM manually."),
    ],
)
def test_gpu_search_feedback_appears_once_in_a_status(
    client: TestClient, query: str, message: str
) -> None:
    html = client.get("/results?gpu_search=" + quote(query)).text
    assert html.count(message) == 1
    status = re.findall(r'<div class="notice" role="status">(.*?)</div>', html, flags=re.S)
    assert sum(message in block for block in status) == 1


def test_gpu_feedback_wording_is_the_protected_text(client: TestClient) -> None:
    ambiguous = client.get("/results?gpu_search=RTX").text
    assert re.search(
        r"GPU search &#34;RTX&#34; matched multiple entries \(.+\)\. "
        r"Select an exact GPU from the list or enter VRAM manually\.",
        ambiguous,
    )
    inferred = client.get("/results?gpu_search=" + quote("RTX 3060 12GB")).text
    assert (
        "GPU &#34;RTX 3060 12GB&#34; was not found. Using 12 GB inferred from your input."
        in inferred
    )
    missing = client.get("/results?gpu_search=Foo").text
    assert "GPU &#34;Foo&#34; was not found. Select a listed GPU or enter VRAM manually." in missing


def test_in_place_refresh_fragment_carries_only_the_current_messages(client: TestClient) -> None:
    clean = client.get("/results/partials/content?vram_gb=24").text
    assert 'role="alert"' not in clean
    assert "Please fix the following input issues:" not in clean
    assert "matched multiple entries" not in clean
    bad = client.get("/results/partials/content?vram_gb=abc").text
    assert bad.count('role="alert"') == 1
    assert 'data-vram-invalid="true"' in bad
    assert 'data-vram-invalid="false"' in clean


def test_full_page_marks_vram_field_invalid_only_when_rejected(client: TestClient) -> None:
    bad = client.get("/results?vram_gb=abc").text
    field = re.search(r'<input\s[^>]*id="vram_gb"[^>]*>', bad, re.S)
    assert field and 'aria-invalid="true"' in field.group(0)
    good = client.get("/results?vram_gb=24").text
    field = re.search(r'<input\s[^>]*id="vram_gb"[^>]*>', good, re.S)
    assert field and "aria-invalid" not in field.group(0)


@pytest.mark.parametrize("url", FORM_PAGES)
def test_native_list_and_free_text_work_without_javascript(client: TestClient, url: str) -> None:
    html = client.get(url).text
    field = re.search(r'<input\s[^>]*id="gpu_search"[^>]*>', html, re.S)
    assert field
    assert 'list="gpu-options"' in field.group(0)
    assert 'type="text"' in field.group(0)
    assert 'role="combobox"' not in html  # added by app.js only
    assert re.findall(
        r'<option data-gpu-id="[^"]+" data-vram-gb="[^"]+" value="([^"]+)"', html
    ) == (_catalog_labels())
    assert 'method="get"' in html and 'action="/results"' in html


@pytest.mark.parametrize("url", FORM_PAGES)
def test_form_wording_follows_the_copy_table(client: TestClient, url: str) -> None:
    html = client.get(url).text
    text = _visible_text(html)
    for gone in REMOVED_FORM_COPY:
        assert gone not in html, gone
    for kept in KEPT_FORM_COPY:
        assert kept in html, kept
    form = html[html.index("<form") : html.index("</form>")]
    assert "How it works" not in _visible_text(form)  # the link now lives only in the header
    assert re.search(r"Filters \(\s*\d+\s*\)", text)
    assert "Clear" in text and "RTX 4060 8GB" in text and "24GB" in text


def test_form_keeps_both_input_paths_and_the_presets(client: TestClient) -> None:
    html = client.get("/results?vram_gb=24&context_tokens=4096").text
    assert 'id="vram_gb"' in html and 'min="1"' in html and 'step="0.1"' in html
    assert len(re.findall(r'name="context_tokens"', html)) == 3
    assert 'id="clear-gpu"' in html
    assert len(re.findall(r'class="chip chip-example"', html)) == 3


def test_styled_list_adds_no_library_and_no_route(client: TestClient) -> None:
    html = client.get("/results?vram_gb=24&context_tokens=4096").text
    scripts = re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
    assert len(scripts) == 2
    assert any("unpkg.com/htmx.org@1.9.12" in s for s in scripts)
    assert any("/static/js/app.js" in s for s in scripts)
    assert 'id="gpu-listbox"' not in html  # built in the browser, not rendered by the server
