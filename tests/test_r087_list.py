"""The verdict list's heading and Models per page (R-087, R-091, ADR-0018 points 3 and 7)."""

from __future__ import annotations

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

REF = "/results?vram_gb=24&context_tokens=4096"


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _get(client: TestClient, url: str) -> str:
    response = client.get(url)
    assert response.status_code == 200, url
    return response.text


def _heading(html: str) -> str:
    return re.search(r'<h2 id="list-heading".*?</h2>', html, re.S).group(0)


def _words(fragment: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", fragment).split())


def test_section_and_heading_markup(client: TestClient) -> None:
    html = _get(client, REF + "&verdict=fits&page=2&page_size=5")
    assert re.search(
        r'<section class="results list-pane pane lite" id="results-cards"'
        r' aria-labelledby="list-heading">\s*<div class="list-head">\s*<h2 id="list-heading"'
        r' tabindex="-1" class="lh-fits">',
        html,
    )
    assert "<b>Fits</b>: showing 6-10 of " in _heading(html)
    assert '<use href="#i-fits"/>' in _heading(html)


@pytest.mark.parametrize(("slug", "icon"), [("tight", "tight"), ("wont-fit", "wont")])
def test_heading_icon_follows_the_verdict(client: TestClient, slug: str, icon: str) -> None:
    html = _get(client, f"/results?vram_gb=8&context_tokens=2048&verdict={slug}")
    assert f'class="lh-{icon}"' in html and f'href="#i-{icon}"' in _heading(html)


def test_single_and_last_item_wording(client: TestClient) -> None:
    html = _get(client, "/results?vram_gb=1&context_tokens=4096&family=yi&verdict=wont-fit")
    assert _words(_heading(html)).endswith("showing 1 of 1 model")
    assert 'class="sizes"' not in html
    total = int(re.search(r"of (\d+) models", _get(client, REF + "&verdict=fits")).group(1))
    last = (total + 4) // 5
    if total % 5 == 1:
        html = _get(client, REF + f"&verdict=fits&page={last}")
        assert f"showing {total} of {total} models" in _words(_heading(html))


def test_sizes_group_and_links(client: TestClient) -> None:
    html = _get(client, REF + "&verdict=fits&page=3&family=llama")
    group = re.search(r'<div class="sizes".*?</div>\s*</div>', html, re.S).group(0)
    assert 'role="group" aria-labelledby="sizes-l"' in group
    assert '<span class="sizes-l" id="sizes-l">Models per page</span>' in group
    links = [htmllib.unescape(a) for a in re.findall(r"<a [^>]*>", group)]
    assert len(links) == 2 and all('data-view-link="size"' in a for a in links)
    assert 'aria-current="true"' in links[0] and "aria-current" not in links[1]
    assert 'href="/results?vram_gb=24&context_tokens=4096&family=llama&verdict=fits"' in links[0]
    assert (
        'href="/results?vram_gb=24&context_tokens=4096&family=llama&verdict=fits&page_size=10"'
        in links[1]
    )
    assert "page=" not in "".join(links).replace("page_size", "")


def test_current_size_stays_a_link_at_10(client: TestClient) -> None:
    html = _get(client, REF + "&verdict=fits&page_size=10")
    links = re.findall(r'<a [^>]*data-view-link="size"[^>]*>(\d+)</a>', html)
    assert links == ["5", "10"]
    assert re.search(r'<a [^>]*aria-current="true">10</a>', html)


def test_no_sizes_at_five_or_fewer_but_shown_at_size_10(client: TestClient) -> None:
    html = _get(client, "/results?vram_gb=8&context_tokens=2048&family=llama&verdict=tight")
    assert html.count("data-model-card") == 4 and 'class="sizes"' not in html
    html = _get(client, "/results?vram_gb=8&context_tokens=2048&verdict=tight&page_size=10")
    assert 'class="sizes"' in html


def test_empty_state_keeps_its_label_and_has_no_heading(client: TestClient) -> None:
    html = _get(client, "/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8")
    assert 'aria-label="Models"' in html and "list-heading" not in html


# ---- page controls (T-044) ----


def _pager(html: str) -> str:
    match = re.search(r'<nav class="pager" aria-label="Pages">.*?</nav>', html, re.S)
    assert match, "no pager"
    return match.group(0)


def _window(html: str) -> str:
    nav = _pager(html)
    items = re.findall(r"<li[^>]*>(.*?)</li>", nav, re.S)
    out = []
    for item in items:
        text = _words(item)
        if text not in ("Previous", "Next"):
            out.append(text)
    return " ".join(out)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (REF + "&verdict=fits", "1 2 … 15"),
        (REF + "&verdict=fits&page=6", "1 … 5 6 7 … 15"),
        (REF + "&verdict=fits&page=15", "1 … 14 15"),
        (REF + "&verdict=fits&page=99", "1 … 14 15"),
        ("/results?vram_gb=1&verdict=wont-fit&page=10", "1 … 9 10 11 … 19"),
    ],
)
def test_page_windows(client: TestClient, url: str, expected: str) -> None:
    assert _window(_get(client, url)) == expected


def test_pager_markup_in_the_middle(client: TestClient) -> None:
    html = _get(client, REF + "&verdict=fits&page=6")
    assert html.index("data-model-card") < html.index('class="pager"')
    assert html.index('class="pager"') < html.index("</section>", html.index("data-model-card"))
    nav = _pager(html)
    assert '<span class="pg pg-cur" aria-current="page">6</span>' in nav
    assert nav.count('<li class="gap" aria-hidden="true">…</li>') == 2
    assert 'aria-label="Page 5"' in nav and 'aria-label="Page 6"' not in nav
    assert 'rel="prev"' in nav and 'rel="next"' in nav
    links = re.findall(r"<a [^>]*>", nav)
    assert len(links) == 6 and all('data-view-link="page"' in a for a in links)


def test_inert_ends_are_spans(client: TestClient) -> None:
    first = _pager(_get(client, REF + "&verdict=fits"))
    assert '<span class="pg pg-off">Previous</span>' in first
    assert 'rel="prev"' not in first and 'rel="next"' in first
    last = _pager(_get(client, REF + "&verdict=fits&page=15"))
    assert '<span class="pg pg-off">Next</span>' in last
    assert 'rel="next"' not in last and 'rel="prev"' in last


def test_page_links_keep_size_and_drop_page_one(client: TestClient) -> None:
    nav = _pager(_get(client, REF + "&verdict=fits&page=2&page_size=10"))
    prev = htmllib.unescape(re.search(r'<a [^>]*rel="prev"[^>]*>', nav).group(0))
    assert f'href="{REF}&verdict=fits&page_size=10"' in prev
    assert "page=1" not in prev.replace("page_size", "")
    nxt = htmllib.unescape(re.search(r'<a [^>]*rel="next"[^>]*>', nav).group(0))
    assert f'href="{REF}&verdict=fits&page=3&page_size=10"' in nxt


def test_no_pager_with_one_page(client: TestClient) -> None:
    html = _get(client, "/results?vram_gb=8&context_tokens=2048&verdict=tight&page_size=10")
    assert 'class="sizes"' in html and "<nav" not in html.split('id="results-cards"')[1]


def test_pager_css_values_are_pinned() -> None:
    from pathlib import Path

    css = (Path(__file__).parents[1] / "src/vramsherpa/static/css/styles.css").read_text()
    for rule in (
        "justify-content: center; align-items: center; gap: 6px; margin: 0; padding: 0; "
        "list-style: none; }",
        "min-height: 44px; min-width: 44px;",
        ".pager .pg-cur { background: var(--pressed);",
        ".pager .pg-off { color: var(--muted); border: 1.5px dashed var(--line); }",
        ".pager a:focus-visible { outline: 3px solid var(--focus);",
    ):
        assert rule in css, rule
