"""Without JavaScript, `Apply filters` keeps the view through one redirect (R-089, ADR-0016)."""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from vramsherpa import database
from vramsherpa.config import Settings
from vramsherpa.main import create_app

BASE = "vram_gb=8&context_tokens=4096"
ZERO_TIGHT = "vram_gb=24&context_tokens=4096&family=qwen"  # no Tight models; Won't fit has 8
PAGE_LOAD_HEADERS = [
    pytest.param({}, id="no-hx-request"),
    pytest.param({"HX-Request": "false"}, id="hx-request-false"),
    pytest.param({"HX-Request": "TRUE"}, id="hx-request-other"),
]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _assert_redirect(response, location: str) -> None:
    assert response.status_code == 302
    assert response.headers["location"] == location
    assert response.content == b""
    assert response.headers.get_list("cache-control") == ["no-cache"]
    vary = [
        name.strip().lower()
        for value in response.headers.get_list("vary")
        for name in value.split(",")
    ]
    assert sorted(vary) == ["hx-history-restore-request", "hx-request"]


def _noscript_button(text: str) -> str:
    [button] = re.findall(r"<noscript>\s*(<button\b[^>]*>Apply filters</button>)", text)
    return button


def _form_submit_buttons(text: str) -> list[str]:
    form = text[text.index('id="search-form"') : text.index("</form>")]
    return re.findall(r'<button\s[^>]*type="submit"[^>]*>', form)


# The button (ADR-0016 point 5).


@pytest.mark.parametrize(
    ("query", "value"),
    [
        (f"{BASE}&verdict=fits", "fits:5"),
        (f"{BASE}&verdict=tight&page=3", "tight:5"),
        (f"{BASE}&verdict=tight&page_size=10", "tight:10"),
        (f"{BASE}&verdict=fits&page_size=7", "fits:5"),
        (f"{ZERO_TIGHT}&verdict=wont-fit&page_size=10", "wont-fit:10"),
    ],
)
def test_button_names_the_view_when_a_valid_verdict_is_in_the_query(
    client: TestClient, query: str, value: str
) -> None:
    response = client.get(f"/results?{query}")

    assert _noscript_button(response.text) == (
        f'<button type="submit" name="apply_view" value="{value}" class="btn-secondary">'
        "Apply filters</button>"
    )


def test_button_names_the_view_when_the_verdict_has_no_models(client: TestClient) -> None:
    response = client.get(f"/results?{ZERO_TIGHT}&verdict=tight&page_size=10")

    assert "No models in Tight with these settings." in response.text
    assert _noscript_button(response.text) == (
        '<button type="submit" name="apply_view" value="tight:10" class="btn-secondary">'
        "Apply filters</button>"
    )


@pytest.mark.parametrize(
    "query", [BASE, f"{BASE}&page_size=10", f"{BASE}&verdict=green", f"{BASE}&verdict="]
)
def test_button_has_no_name_without_a_valid_verdict(client: TestClient, query: str) -> None:
    response = client.get(f"/results?{query}")

    assert _noscript_button(response.text) == (
        '<button type="submit" class="btn-secondary">Apply filters</button>'
    )
    assert "apply_view" not in response.text


def test_check_compatibility_stays_first_with_no_name(client: TestClient) -> None:
    buttons = _form_submit_buttons(client.get(f"/results?{BASE}&verdict=fits").text)

    assert buttons[0] == '<button type="submit" class="btn-primary">'
    assert 'name="apply_view"' in buttons[1]
    assert len(buttons) == 2


def test_form_has_autocomplete_off(client: TestClient) -> None:
    text = client.get(f"/results?{BASE}").text
    form_tag = text[text.rindex("<form", 0, text.index('id="search-form"')) :].split(">", 1)[0]

    assert 'autocomplete="off"' in form_tag


# The redirect (ADR-0016 point 6).


@pytest.mark.parametrize("headers", PAGE_LOAD_HEADERS)
@pytest.mark.parametrize(
    ("query", "location"),
    [
        (f"{BASE}&apply_view=fits:10", f"/results?{BASE}&verdict=fits&page_size=10"),
        (f"{BASE}&apply_view=fits:5", f"/results?{BASE}&verdict=fits"),
        (f"{BASE}&apply_view=wont-fit%3A10", f"/results?{BASE}&verdict=wont-fit&page_size=10"),
        (f"apply_view=tight%3a5&{BASE}", f"/results?{BASE}&verdict=tight"),
        (f"{BASE}&%61pply_view=tight:5", f"/results?{BASE}&verdict=tight"),
    ],
)
def test_apply_view_redirects_to_the_real_parameters(
    client: TestClient, query: str, location: str, headers: dict[str, str]
) -> None:
    response = client.get(f"/results?{query}", headers=headers, follow_redirects=False)

    _assert_redirect(response, location)


def test_apply_view_replaces_any_view_in_the_query_and_drops_page(client: TestClient) -> None:
    query = f"{BASE}&verdict=fits&page=3&page_size=5&apply_view=tight:10"

    response = client.get(f"/results?{query}", follow_redirects=False)

    _assert_redirect(response, f"/results?{BASE}&verdict=tight&page_size=10")


@pytest.mark.parametrize(
    "raw",
    [
        "apply_view=green:5",
        "apply_view=fits:7",
        "apply_view=",
        "apply_view",
        "apply_view=fits",
        "apply_view=FITS:5",
        "apply_view=fits:5:5",
        "apply_view=fits:10&apply_view=green:5",
    ],
)
def test_invalid_apply_view_is_dropped_and_gives_the_gist(client: TestClient, raw: str) -> None:
    query = f"{BASE}&verdict=fits&page=2&page_size=10&{raw}"

    response = client.get(f"/results?{query}", follow_redirects=False)

    _assert_redirect(response, f"/results?{BASE}")


def test_repeated_apply_view_uses_the_last_value(client: TestClient) -> None:
    query = f"{BASE}&apply_view=green:5&apply_view=fits:7&apply_view=tight:10"

    response = client.get(f"/results?{query}", follow_redirects=False)

    _assert_redirect(response, f"/results?{BASE}&verdict=tight&page_size=10")


def test_encoded_query_passes_through_unchanged(client: TestClient) -> None:
    kept = (
        "gpu_id=&gpu_search=RTX%204090%20%26%2B%3D%25+x&vram_gb="
        "&family=a&family=b&context_tokens=2048&note=%0D%0Aset-cookie%3Ax"
    )

    response = client.get(f"/results?{kept}&apply_view=fits:10", follow_redirects=False)

    _assert_redirect(response, f"/results?{kept}&verdict=fits&page_size=10")


def test_location_is_relative_whatever_the_host_and_scheme(client: TestClient) -> None:
    response = client.get(
        f"https://localhost/results?{BASE}&next=//evil.example.com&apply_view=fits:5",
        headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "evil.example.com"},
        follow_redirects=False,
    )

    _assert_redirect(response, f"/results?{BASE}&next=//evil.example.com&verdict=fits")


def test_following_the_redirect_shows_the_list_and_the_same_button(client: TestClient) -> None:
    # What a browser sends: every field, the colon encoded, and no `verdict` or `page_size`.
    form = "gpu_id=&gpu_search=&vram_gb=8&context_tokens=4096&min_params_b=&max_params_b="

    response = client.get(f"/results?{form}&apply_view=tight%3A10")

    assert [r.status_code for r in response.history] == [302]
    assert response.request.url.query.decode() == f"{form}&verdict=tight&page_size=10"
    assert "<b>Tight</b>: showing 1-10 of 13 models" in response.text
    assert 'value="tight:10"' in _noscript_button(response.text)


# Order: no hardware first, then this, then the database.


@pytest.mark.parametrize(
    "query", ["apply_view=fits:5", "apply_view=green:5", "gpu_search=%20&apply_view=fits:10"]
)
def test_no_hardware_still_goes_to_landing(client: TestClient, query: str) -> None:
    response = client.get(f"/results?{query}", follow_redirects=False)

    _assert_redirect(response, "/")


@pytest.mark.parametrize("value", ["fits:10", "green:5"])
def test_redirect_makes_no_database_query(client: TestClient, value: str) -> None:
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    engine = database.get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.get(f"/results?{BASE}&apply_view={value}", follow_redirects=False)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert response.status_code == 302
    assert statements == []


def test_malformed_context_keeps_the_422(client: TestClient) -> None:
    response = client.get(
        "/results?vram_gb=8&context_tokens=abc&apply_view=fits:5", follow_redirects=False
    )

    assert response.status_code == 422
    assert "location" not in response.headers


@pytest.mark.parametrize("app_env", ["production", "test"])
def test_unknown_host_gets_400_json_without_location(test_database_url: str, app_env: str) -> None:
    strict = create_app(
        Settings(database_url=None, app_env=app_env, allowed_hosts=("testserver",)),
        database_url=test_database_url,
    )

    response = TestClient(strict).get(
        f"/results?{BASE}&apply_view=fits:5",
        headers={"Host": "evil.example.com"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Host not allowed"}
    assert "location" not in response.headers


# `apply_view` is never written out.


@pytest.mark.parametrize(
    "path",
    [
        "/results",
        "/results/partials/content",
        "/results/partials/summary",
        "/results/partials/list",
    ],
)
@pytest.mark.parametrize("view", ["", "&verdict=tight&page_size=10"])
def test_htmx_request_is_not_redirected_and_writes_no_apply_view(
    client: TestClient, path: str, view: str
) -> None:
    response = client.get(
        f"{path}?{BASE}{view}&apply_view=fits:5",
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "location" not in response.headers
    assert "apply_view" not in response.text
    assert "apply_view" not in response.headers.get("hx-push-url", "")


def test_pushed_address_drops_only_apply_view_byte_for_byte(client: TestClient) -> None:
    kept = "gpu_search=RTX%204090&vram_gb=&family=a&&verdict=tight&page=2"

    response = client.get(
        f"/results/partials/content?{kept}&%61pply_view=fits:5&page_size=10",
        headers={"HX-Request": "true"},
    )

    assert response.headers["hx-push-url"] == f"/results?{kept}&page_size=10"


def test_history_restore_full_page_writes_no_apply_view_link(client: TestClient) -> None:
    response = client.get(
        f"/results?{BASE}&verdict=tight&apply_view=fits:5",
        headers={"HX-Request": "true", "HX-History-Restore-Request": "true"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert response.text.count("apply_view") == 1
    assert 'name="apply_view" value="tight:5"' in response.text


def test_old_partial_link_with_apply_view_takes_two_relative_hops(client: TestClient) -> None:
    response = client.get(f"/results/partials/content?{BASE}&apply_view=fits:10")

    assert [r.headers["location"] for r in response.history] == [
        f"/results?{BASE}&apply_view=fits:10",
        f"/results?{BASE}&verdict=fits&page_size=10",
    ]
    assert response.status_code == 200
