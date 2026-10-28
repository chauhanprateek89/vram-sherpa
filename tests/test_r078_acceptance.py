"""QA acceptance checks for R-078 (behaviour the overhaul keeps). Black-box, over HTTP.

The figures and order of the two reference queries are guarded by test_reference_queries.py. These
checks cover the rest of R-078 that the server's HTML can show: the copy table (Old strings gone,
New strings present), protected text kept word for word, routes and form fields unchanged, the
three figures on every verdict (rule 16) and the no-JavaScript markup. Behaviour that needs a
browser (live refresh, the quantisation switcher, JavaScript off) is in .sdlc/qa/R-078/report.md.
"""

from __future__ import annotations

import html as htmllib
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

REF = "/results?vram_gb=24&context_tokens=4096"
REF2 = "/results?vram_gb=12&context_tokens=8192&family=qwen&quant_bucket=Q4"
EMPTY = "/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8"


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _text(markup: str) -> str:
    markup = re.sub(r"<(script|style)\b.*?</\1>", " ", markup, flags=re.S)
    markup = re.sub(r"<[^>]+>", " ", markup)
    markup = re.sub(r"\s+", " ", htmllib.unescape(markup))
    # Tags between "(" and the figure leave stray spaces: "Filters ( 0 )" reads "Filters (0)".
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", markup))


def _body(client: TestClient, url: str) -> str:
    response = client.get(url)
    assert response.status_code == 200
    return response.text


@pytest.mark.parametrize("url", [REF, "/results?gpu_id=no_such_gpu"])
def test_copy_table_old_strings_are_gone(client: TestClient, url: str) -> None:
    page = _text(_body(client, url))
    page_raw = _body(client, url)

    for old in [
        "Choose either your exact GPU or enter available VRAM directly.",
        "I know my GPU",
        "I know my VRAM",
        "If your GPU is not listed, leave this blank and enter VRAM instead.",
        "Lower bound on model size in billions of parameters.",
        "Upper bound on model size in billions of parameters.",
        "Summary for this query",
        "Showing only Fits and Tight models in the list below.",
        "Highest-ranked variant that remains in the Fits band.",
        "Next highest-ranked practical option by current ranking order.",
        "Lowest-ranked viable option, usually the lightest footprint.",
    ]:
        assert old not in page, old
    assert "(or leave blank)" not in page_raw
    # The in-form "How it works" link is removed; the header keeps the only one.
    assert page.count("How it works") == 1


def test_copy_table_new_strings_are_present(client: TestClient) -> None:
    page = _text(_body(client, REF))

    for new in [
        "Pick your GPU, or enter your VRAM if it isn't listed.",
        "71 models fit or are tight",
        "Only Fits and Tight models are listed. Pick a quantisation on any card to see how it "
        "changes.",
        "The strongest model that still leaves room to spare.",
        "Nearly as capable, and comfortable to run.",
        "The lightest model that still runs on your GPU.",
        "Loading the breakdown",
        "Open to load the breakdown.",
        "Filters (0)",
    ]:
        assert new in page, new
    assert 'placeholder="Start typing a GPU name"' in _body(client, REF)


def test_empty_state_wording(client: TestClient) -> None:
    page = _text(_body(client, EMPTY))

    assert "0 models fit or are tight" in page
    assert "Only Fits and Tight models are listed." in page
    assert "Pick a quantisation on any card" not in page
    assert "There are 95 models in Won't fit, hidden by default." in page
    assert "variants in Won't fit" not in page
    assert "No models currently fit this setup." in page
    assert "Try one of these next steps:" in page


def test_breakdown_wording(client: TestClient) -> None:
    variant = "variant_model_deepseek_coder_v2_lite_instruct_q4"
    response = client.get(
        f"/results/why/{variant}?context_tokens=4096&vram_gb=24", headers={"HX-Request": "true"}
    )
    page = _text(response.text)

    assert response.status_code == 200
    for label, value in [
        ("Model weights", "9.72 GB"),
        ("Context memory (KV cache)", "3.85 GB"),
        ("Runtime overhead", "1.79 GB"),
        ("Total needed", "15.36 GB"),
        ("Safety reserve", "2.40 GB"),
    ]:
        assert f"{label}" in page and value in page, label
    assert "Kept free, not part of the total" in page
    assert page.index("Total needed") < page.index("Safety reserve")
    for raw in ["weights_gb:", "kv_cache_gb:", "reserve_gb:", "required_vram_gb:"]:
        assert raw not in page


def test_protected_text_is_kept(client: TestClient) -> None:
    page = _text(_body(client, REF))
    for kept in [
        "Search GPU",
        "Clear",
        "Available VRAM (GB)",
        "Context window",
        "2k",
        "4k",
        "8k",
        "RTX 4060 8GB",
        "8GB",
        "24GB",
        "Check compatibility",
        "Min params (B)",
        "Max params (B)",
        "Recommended only",
        "Apply filters",
        "Hardware: Manual VRAM (24 GB)",
        "Context: 4k",
        "Top Picks for you",
        "Best quality that fits",
        "Best balance",
        "Best small",
        "How this is estimated",
        "Catalog version: 2026-10-23",
        "Estimation policy: v0.1",
        "Last updated: 2026-10-23",
    ]:
        assert kept in page, kept
    for label in ["Fits", "Tight", "Won't fit"]:
        assert label in page

    errors = _text(_body(client, "/results?vram_gb=abc"))
    assert "Please fix the following input issues:" in errors
    assert "Invalid value for vram_gb." in errors

    chips = _text(_body(client, REF2 + "&min_params_b=8&max_params_b=8"))
    for chip in ["Family: qwen", "Quant: Q4", "Min params >= 8B", "Max params <= 8B"]:
        assert chip in chips, chip


def test_gpu_search_messages_word_for_word(client: TestClient) -> None:
    ambiguous = _text(_body(client, "/results?gpu_search=RTX"))
    assert 'GPU search "RTX" matched multiple entries (' in ambiguous
    assert "Select an exact GPU from the list or enter VRAM manually." in ambiguous
    assert ambiguous.count('GPU search "RTX" matched multiple entries') == 1

    inferred = _text(_body(client, "/results?gpu_search=RTX 3060 12GB"))
    assert 'GPU "RTX 3060 12GB" was not found. Using 12 GB inferred from your input.' in inferred

    default = _text(_body(client, "/results?gpu_id=no_such_gpu"))
    assert (
        "No GPU or VRAM selected yet. Showing estimates with an assumed 8 GB baseline." in default
    )


def test_every_verdict_pane_shows_required_available_and_margin(client: TestClient) -> None:
    """Rule 16: each variant pane carries its verdict, needed, available and margin figures."""
    for url in [REF, REF2, "/results?vram_gb=2&context_tokens=4096"]:
        markup = _body(client, url)
        panes = re.split(r"data-variant-pane=", markup)[1:]
        assert panes, url
        for pane in panes:
            pane = re.split(r"</article>", pane)[0]
            text = _text(pane)
            gauge = re.search(
                r"Needs ([\d.]+) of ([\d.]+) gigabytes, with ([\d.]+) gigabytes held in reserve",
                pane,
            )
            assert gauge, pane[:300]
            assert re.search(r"\b(Fits|Tight|Won't fit)\b", text), text
            assert f"{gauge.group(1)} GB needed" in text, text
            assert re.search(rf"(?<![\d.]){re.escape(gauge.group(2))} GB", text), text
            assert re.search(r"[\d.]+ GB (spare|over)", text), text


def test_routes_and_fragments_are_kept(client: TestClient) -> None:
    query = "vram_gb=24&context_tokens=4096"
    for path in ["/", "/how-it-works", "/healthz", REF, "/models/model_llama_31_8b_instruct"]:
        assert client.get(path).status_code == 200, path
    for fragment in ["summary", "list"]:
        assert client.get(f"/results/partials/{fragment}?{query}").status_code == 200
    assert client.get("/models/nonexistent").status_code == 404

    htmx = client.get(f"/results/partials/content?{query}", headers={"HX-Request": "true"})
    assert htmx.status_code == 200
    # The one agreed exception (R-081): an old fragment URL opened as a page redirects.
    page = client.get(f"/results/partials/content?{query}", follow_redirects=False)
    assert page.status_code == 302
    assert page.headers["location"] == f"/results?{query}"


def test_form_fields_and_steps_are_kept(client: TestClient) -> None:
    markup = _body(client, REF)

    assert 'id="search-form"' in markup and 'action="/results"' in markup
    assert 'hx-get="/results/partials/content"' in markup
    assert 'hx-target="#results-shell"' in markup
    for name in ["gpu_id", "gpu_search", "vram_gb", "context_tokens", "family", "quant_bucket"]:
        assert f'name="{name}"' in markup, name
    for name in ["min_params_b", "max_params_b", "recommended_only"]:
        assert f'name="{name}"' in markup, name
    vram = re.search(r"<input[^>]*id=\"vram_gb\"[^>]*>", markup, re.S)
    assert vram and 'min="1"' in vram.group(0) and 'step="0.1"' in vram.group(0)
    for value in ["2048", "4096", "8192"]:
        assert f'name="context_tokens" value="{value}"' in markup or re.search(
            rf'<input[^>]*name="context_tokens"[^>]*value="{value}"', markup, re.S
        )
    assert markup.count('<option value="gpu_') >= 35 or markup.count("<option") >= 35


def test_no_javascript_markup(client: TestClient) -> None:
    """With JavaScript off: Apply filters is a real button, and collapsed sections are <details>."""
    markup = _body(client, REF)

    assert re.search(r"<noscript>\s*<button[^>]*type=\"submit\"[^>]*>\s*Apply filters", markup)
    filters = re.search(r"<details class=\"filters[^>]*>", markup)
    assert filters and "hidden" not in filters.group(0)
    picks = re.search(r"<details class=\"picks[^>]*>", markup)
    assert picks and "hidden" not in picks.group(0)
    assert markup.count('<details class="why-panel"') >= 70
    assert 'method="get"' in markup
    # Server-rendered gauge widths, no script needed.
    assert re.search(r'class="bar bar-fits" style="width: [\d.]+%;"', markup)
