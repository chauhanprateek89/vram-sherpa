"""Guards for htmx history entries, history misses and breakdown URLs (R-081, ADR-0008)."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from vramsherpa import database
from vramsherpa.main import _results_page_url

QUERY = "vram_gb=24&context_tokens=4096"
HX = {"HX-Request": "true"}
RESTORE = {"HX-Request": "true", "HX-History-Restore-Request": "true"}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}

# Full pages rendered from base.html, each of which must carry the htmx-config meta tag.
FULL_PAGE_URLS = [
    "/",
    f"/results?{QUERY}",
    "/models/model_qwen25_7b_instruct",
    "/how-it-works",
]


@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    text: str = ""

    def ancestors(self) -> list[Node]:
        found: list[Node] = []
        node = self.parent
        while node is not None:
            found.append(node)
            node = node.parent
        return found

    def descendants(self) -> list[Node]:
        found: list[Node] = []
        for child in self.children:
            found.append(child)
            found.extend(child.descendants())
        return found


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root = Node("#document", {})
        self.current = self.root
        self.nodes: list[Node] = []

    def _add(self, tag: str, attrs: list[tuple[str, str | None]]) -> Node:
        node = Node(tag, {name: value or "" for name, value in attrs}, parent=self.current)
        self.current.children.append(node)
        self.nodes.append(node)
        return node

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = self._add(tag, attrs)
        if tag not in VOID_ELEMENTS:
            self.current = node

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._add(tag, attrs)

    def handle_data(self, data: str) -> None:
        self.current.text += data

    def handle_endtag(self, tag: str) -> None:
        node = self.current
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.current = node.parent


def _parse(html: str) -> _TreeBuilder:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder


def _by_id(tree: _TreeBuilder, element_id: str) -> list[Node]:
    return [node for node in tree.nodes if node.attrs.get("id") == element_id]


def _count(tree: _TreeBuilder, tag: str) -> int:
    return sum(1 for node in tree.nodes if node.tag == tag)


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


# Fix 1 and fix 4's shared helper: one place builds the /results?<query> string.


def test_results_page_url_passes_the_query_through_unchanged() -> None:
    query = "vram_gb=24&context_tokens=4096&family=a&family=b&gpu_search=RTX%204090%20%2824GB%29"
    assert _results_page_url(query) == f"/results?{query}"


def test_results_page_url_without_a_query_is_the_bare_path() -> None:
    assert _results_page_url("") == "/results"


# Fix 1: the partial pushes the full-page URL.


def test_partial_content_pushes_full_page_url_with_repeated_params_in_order(
    client: TestClient,
) -> None:
    query = "vram_gb=24&context_tokens=4096&family=a&family=b"
    response = client.get(f"/results/partials/content?{query}", headers=HX)

    assert response.status_code == 200
    assert response.headers["HX-Push-Url"] == f"/results?{query}"
    assert 'id="results-summary-shell"' in response.text
    assert 'id="search-form"' not in response.text


def test_partial_content_push_url_keeps_percent_encoding_byte_for_byte(
    client: TestClient,
) -> None:
    query = (
        "gpu_search=NVIDIA%20GeForce%20RTX%204090%20%2824.0%20GB%29"
        "&context_tokens=2048&quant_bucket=Q8&quant_bucket=Q4"
    )
    response = client.get(f"/results/partials/content?{query}", headers=HX)

    assert response.status_code == 200
    assert response.headers["HX-Push-Url"] == f"/results?{query}"


def test_partial_content_without_query_pushes_bare_results(client: TestClient) -> None:
    response = client.get("/results/partials/content", headers=HX)

    assert response.status_code == 200
    assert response.headers["HX-Push-Url"] == "/results"


# Fix 3: a history-restore request to /results gets the full page.


def test_results_history_restore_request_returns_full_page(client: TestClient) -> None:
    response = client.get(f"/results?{QUERY}", headers=RESTORE)

    assert response.status_code == 200
    tree = _parse(response.text)
    assert _count(tree, "header") == 1
    assert len(_by_id(tree, "search-form")) == 1
    assert _count(tree, "footer") == 1


def test_results_htmx_request_without_restore_still_returns_fragment(client: TestClient) -> None:
    response = client.get(f"/results?{QUERY}", headers=HX)

    assert response.status_code == 200
    tree = _parse(response.text)
    assert _count(tree, "header") == 0
    assert _by_id(tree, "search-form") == []
    assert _count(tree, "footer") == 0
    assert len(_by_id(tree, "results-summary-shell")) == 1


def test_results_restore_header_alone_returns_full_page(client: TestClient) -> None:
    response = client.get(f"/results?{QUERY}", headers={"HX-History-Restore-Request": "true"})

    assert response.status_code == 200
    assert len(_by_id(_parse(response.text), "search-form")) == 1


# Fix 2: a history miss reloads the page.


@pytest.mark.parametrize("url", FULL_PAGE_URLS)
def test_full_pages_set_refresh_on_history_miss_before_htmx(client: TestClient, url: str) -> None:
    response = client.get(url)
    assert response.status_code == 200

    tree = _parse(response.text)
    head = next(node for node in tree.nodes if node.tag == "head")
    in_head = head.descendants()
    metas = [
        node for node in in_head if node.tag == "meta" and node.attrs.get("name") == "htmx-config"
    ]
    assert len(metas) == 1
    assert metas[0].attrs["content"] == '{"refreshOnHistoryMiss":true}'

    htmx_scripts = [
        node for node in in_head if node.tag == "script" and "htmx.org" in node.attrs.get("src", "")
    ]
    assert len(htmx_scripts) == 1
    assert in_head.index(metas[0]) < in_head.index(htmx_scripts[0])


# ADR-0006 point 9: the results region sits after the form, so breakdowns push no URL.


def test_results_shell_sits_after_the_search_form(client: TestClient) -> None:
    response = client.get(f"/results?{QUERY}")
    tree = _parse(response.text)

    [form] = _by_id(tree, "search-form")
    [shell] = _by_id(tree, "results-shell")
    assert form not in shell.ancestors()
    assert shell not in form.descendants()
    assert tree.nodes.index(shell) > tree.nodes.index(form) + len(form.descendants())


def test_breakdown_triggers_have_no_push_url_ancestor(client: TestClient) -> None:
    response = client.get(f"/results?{QUERY}")
    tree = _parse(response.text)

    triggers = [
        node for node in tree.nodes if node.attrs.get("hx-get", "").startswith("/results/why/")
    ]
    assert triggers, "the reference query should render at least one breakdown"
    for trigger in triggers:
        assert "hx-push-url" not in trigger.attrs
        pushing = [node.tag for node in trigger.ancestors() if "hx-push-url" in node.attrs]
        assert pushing == [], f"{trigger.attrs['hx-get']} inherits hx-push-url from {pushing}"


# Fix 4: old /results/partials/content links opened as a page redirect to /results.

OLD_LINK_QUERY = "vram_gb=24&context_tokens=4096&family=a&family=b"
PAGE_LOAD_HEADERS = [
    pytest.param({}, id="no-hx-request"),
    pytest.param({"HX-Request": "false"}, id="hx-request-false"),
]
PARTIAL_PATHS = [
    "/results/partials/content",
    "/results/partials/summary",
    "/results/partials/list",
]
# A marker each partial's fragment renders for the reference query.
PARTIAL_MARKERS = {
    "/results/partials/content": 'id="results-summary-shell"',
    "/results/partials/summary": 'id="summary-banner"',
    "/results/partials/list": 'class="sr-only">Results<',
}


def _vary_names(response) -> list[str]:
    return [
        name.strip().lower()
        for value in response.headers.get_list("vary")
        for name in value.split(",")
        if name.strip()
    ]


def _assert_old_link_redirect(response, location: str) -> None:
    assert response.status_code == 302
    assert response.headers["location"] == location
    assert response.content == b""
    assert response.headers.get_list("cache-control") == ["no-cache"]
    assert sorted(_vary_names(response)) == ["hx-history-restore-request", "hx-request"]


@pytest.mark.parametrize("headers", PAGE_LOAD_HEADERS)
def test_old_partial_link_redirects_to_results_with_query_byte_for_byte(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get(
        f"/results/partials/content?{OLD_LINK_QUERY}", headers=headers, follow_redirects=False
    )

    _assert_old_link_redirect(response, f"/results?{OLD_LINK_QUERY}")


def test_old_partial_link_keeps_percent_encoding_in_location(client: TestClient) -> None:
    query = (
        "gpu_search=NVIDIA%20GeForce%20RTX%204090%20%2824.0%20GB%29"
        "&context_tokens=2048&quant_bucket=Q8&quant_bucket=Q4&gpu_search=a%2Bb+c"
    )
    response = client.get(f"/results/partials/content?{query}", follow_redirects=False)

    _assert_old_link_redirect(response, f"/results?{query}")


@pytest.mark.parametrize("headers", PAGE_LOAD_HEADERS)
def test_old_partial_link_without_query_redirects_to_bare_results(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/results/partials/content", headers=headers, follow_redirects=False)

    _assert_old_link_redirect(response, "/results")


def test_old_partial_link_location_is_a_path_whatever_the_host_or_scheme(
    client: TestClient,
) -> None:
    response = client.get(
        f"https://localhost/results/partials/content?{QUERY}",
        headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "evil.example.com"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == f"/results?{QUERY}"


def test_old_partial_link_redirect_lands_on_the_full_page(client: TestClient) -> None:
    response = client.get(f"/results/partials/content?{QUERY}")

    assert response.status_code == 200
    assert [r.status_code for r in response.history] == [302]
    assert str(response.url) == f"http://testserver/results?{QUERY}"
    tree = _parse(response.text)
    assert _count(tree, "header") == 1
    assert len(_by_id(tree, "search-form")) == 1
    assert _count(tree, "footer") == 1


def test_old_partial_link_redirect_makes_no_database_query(client: TestClient) -> None:
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    engine = database.get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.get(f"/results/partials/content?{QUERY}", follow_redirects=False)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert response.status_code == 302
    assert statements == []


@pytest.mark.parametrize("headers", PAGE_LOAD_HEADERS)
def test_old_partial_link_with_malformed_value_gets_the_same_422_as_results(
    client: TestClient, headers: dict[str, str]
) -> None:
    expected = client.get("/results?context_tokens=abc", follow_redirects=False)
    response = client.get(
        "/results/partials/content?context_tokens=abc", headers=headers, follow_redirects=False
    )

    assert expected.status_code == 422
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == expected.json()
    assert "location" not in response.headers


@pytest.mark.parametrize("path", PARTIAL_PATHS)
def test_partials_with_hx_request_return_their_fragment_without_location(
    client: TestClient, path: str
) -> None:
    response = client.get(f"{path}?{QUERY}", headers=HX, follow_redirects=False)

    assert response.status_code == 200
    assert "location" not in response.headers
    assert PARTIAL_MARKERS[path] in response.text
    assert 'id="search-form"' not in response.text


@pytest.mark.parametrize("path", ["/results/partials/summary", "/results/partials/list"])
@pytest.mark.parametrize("headers", PAGE_LOAD_HEADERS)
def test_summary_and_list_partials_never_redirect(
    client: TestClient, path: str, headers: dict[str, str]
) -> None:
    response = client.get(f"{path}?{QUERY}", headers=headers, follow_redirects=False)

    assert response.status_code == 200
    assert "location" not in response.headers
    assert PARTIAL_MARKERS[path] in response.text


def test_partial_content_restore_request_still_gets_the_fragment(client: TestClient) -> None:
    response = client.get(
        f"/results/partials/content?{QUERY}", headers=RESTORE, follow_redirects=False
    )

    assert response.status_code == 200
    assert "location" not in response.headers
    assert 'id="results-summary-shell"' in response.text


def test_old_partial_link_with_unknown_host_is_rejected_before_redirect(
    client: TestClient,
) -> None:
    response = client.get(
        "/results/partials/content?vram_gb=24",
        headers={"Host": "evil.example.com"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Host not allowed"}
    assert "location" not in response.headers


# R-078, ADR-0006 point 1: the frozen form interface.


@pytest.mark.parametrize("url", ["/results?gpu_id=no_such_gpu", f"/results?{QUERY}"])
def test_search_form_interface_is_unchanged(client: TestClient, url: str) -> None:
    tree = _parse(client.get(url).text)

    [form] = _by_id(tree, "search-form")
    assert form.tag == "form"
    assert form.attrs["method"] == "get"
    assert form.attrs["action"] == "/results"
    assert form.attrs["hx-get"] == "/results/partials/content"
    assert form.attrs["hx-target"] == "#results-shell"
    assert form.attrs["hx-push-url"] == "true"
    assert form.attrs["hx-trigger"] == (
        "submit, change from:.js-auto-filter, keyup changed delay:350ms from:.js-auto-filter-text"
    )

    in_form = form.descendants()
    noscripts = [node for node in in_form if node.tag == "noscript"]
    assert len(noscripts) == 1
    buttons = [node for node in noscripts[0].descendants() if node.tag == "button"]
    assert [(b.attrs.get("type"), b.text.strip()) for b in buttons] == [("submit", "Apply filters")]

    form_ids = {node.attrs.get("id") for node in in_form}
    assert {"gpu_search", "gpu_id", "gpu-options", "clear-gpu", "vram_gb"} <= form_ids
    assert any("chip-example" in node.attrs.get("class", "").split() for node in in_form)
    for name in ("family", "quant_bucket", "context_tokens", "recommended_only"):
        assert any(node.attrs.get("name") == name for node in in_form), name
    assert len(_by_id(tree, "theme-toggle")) == 1
    assert len(_by_id(tree, "results-shell")) == 1
