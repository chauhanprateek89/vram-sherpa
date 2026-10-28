"""Guards for the redirect of a hardware-less `/results` page load to `/` (R-013, ADR-0010)."""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from vramsherpa import database, main
from vramsherpa.config import Settings
from vramsherpa.main import _has_hardware_input, create_app

DEFAULT_NOTE = "No GPU or VRAM selected yet. Showing estimates with an assumed 8 GB baseline."
ASSUMED_8GB = "Assumed VRAM (8 GB)"

NO_HARDWARE_URLS = [
    "/results",
    "/results?context_tokens=4096&family=qwen",
    "/results?gpu_search=%20%20&vram_gb=",
    "/results?gpu_id=&gpu_search=&vram_gb=&context_tokens=2048",
]
PAGE_LOAD_HEADERS = [
    pytest.param({}, id="no-hx-request"),
    pytest.param({"HX-Request": "false"}, id="hx-request-false"),
    pytest.param({"HX-Request": "TRUE"}, id="hx-request-other"),
]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _vary_names(response) -> list[str]:
    return [
        name.strip().lower()
        for value in response.headers.get_list("vary")
        for name in value.split(",")
        if name.strip()
    ]


def _assert_redirect_to_landing(response) -> None:
    assert response.status_code == 302
    assert response.headers["location"] == "/"
    assert response.content == b""
    assert response.headers.get_list("cache-control") == ["no-cache"]
    assert sorted(_vary_names(response)) == ["hx-history-restore-request", "hx-request"]


# Point 2: one helper states "no hardware" once.


@pytest.mark.parametrize(
    ("gpu_id", "gpu_search", "vram_gb"),
    [
        (None, None, None),
        ("", "", ""),
        ("  ", "\t", " \n "),
    ],
)
def test_helper_sees_no_hardware_in_absent_or_blank_values(gpu_id, gpu_search, vram_gb) -> None:
    assert _has_hardware_input(gpu_id, gpu_search, vram_gb) is False


@pytest.mark.parametrize(
    ("gpu_id", "gpu_search", "vram_gb"),
    [
        ("no_such_gpu", None, None),
        (None, " RTX ", None),
        (None, None, "abc"),
        (None, "", "0"),
    ],
)
def test_helper_sees_any_non_blank_hardware_value(gpu_id, gpu_search, vram_gb) -> None:
    assert _has_hardware_input(gpu_id, gpu_search, vram_gb) is True


# Points 1 to 3: the redirect itself.


@pytest.mark.parametrize("headers", PAGE_LOAD_HEADERS)
@pytest.mark.parametrize("url", NO_HARDWARE_URLS)
def test_page_load_without_hardware_redirects_to_landing(
    client: TestClient, url: str, headers: dict[str, str]
) -> None:
    response = client.get(url, headers=headers, follow_redirects=False)

    _assert_redirect_to_landing(response)


def test_redirect_ignores_every_other_parameter(client: TestClient) -> None:
    url = (
        "/results?context_tokens=8192&family=qwen&family=llama&quant_bucket=Q4"
        "&min_params_b=1&max_params_b=70&recommended_only=true"
    )
    response = client.get(url, follow_redirects=False)

    _assert_redirect_to_landing(response)


def test_redirect_location_ignores_host_and_scheme(client: TestClient) -> None:
    response = client.get(
        "https://localhost/results?context_tokens=4096",
        headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "evil.example.com"},
        follow_redirects=False,
    )

    _assert_redirect_to_landing(response)


def test_redirect_makes_no_database_query(client: TestClient) -> None:
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    engine = database.get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.get("/results?context_tokens=4096", follow_redirects=False)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert response.status_code == 302
    assert statements == []


def test_redirect_is_checked_first_in_the_handler_through_the_helper() -> None:
    body = inspect.getsource(main.results).split(") -> Response:", 1)[1]
    first_statement = next(
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert first_statement.startswith("if not _has_hardware_input(gpu_id, gpu_search, vram_gb)")
    assert 'RedirectResponse(\n            "/",' in body
    assert "status_code=302" in body


def test_redirect_ends_on_landing_page(client: TestClient) -> None:
    response = client.get("/results?gpu_search=")

    assert response.status_code == 200
    assert [r.status_code for r in response.history] == [302]
    assert str(response.url) == "http://testserver/"


# Point 4: what is not redirected.


def test_htmx_request_without_hardware_gets_fragment_and_default_note(
    client: TestClient,
) -> None:
    response = client.get("/results", headers={"HX-Request": "true"}, follow_redirects=False)

    assert response.status_code == 200
    assert "location" not in response.headers
    assert response.text.lstrip().startswith('<div id="results-summary-shell">')
    assert "<header" not in response.text
    assert DEFAULT_NOTE in response.text


def test_history_restore_without_hardware_gets_the_full_page(client: TestClient) -> None:
    response = client.get(
        "/results",
        headers={"HX-Request": "true", "HX-History-Restore-Request": "true"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "location" not in response.headers
    assert response.text.count("<header") == 1
    assert 'id="search-form"' in response.text
    assert response.text.count("<footer") == 1
    assert DEFAULT_NOTE in response.text


@pytest.mark.parametrize(
    ("url", "error"),
    [
        ("/results?vram_gb=abc", "Invalid value for vram_gb."),
        ("/results?vram_gb=0", "vram_gb must be greater than or equal to 1e-06."),
        ("/results?gpu_id=no_such_gpu", None),
    ],
)
def test_unusable_hardware_still_renders_results_at_8gb(
    client: TestClient, url: str, error: str | None
) -> None:
    response = client.get(url, follow_redirects=False)

    assert response.status_code == 200
    assert "location" not in response.headers
    assert ASSUMED_8GB in response.text
    assert DEFAULT_NOTE in response.text
    assert "model-card" in response.text
    if error is None:
        assert "Please fix the following input issues:" not in response.text
    else:
        assert response.text.count(error) == 1


def test_bad_context_without_hardware_keeps_the_422(client: TestClient) -> None:
    response = client.get("/results?context_tokens=abc", follow_redirects=False)

    assert response.status_code == 422
    assert "location" not in response.headers


# Rule 8 and rule 18: the host guard runs before the redirect.


def test_unknown_host_gets_400_json_without_location(test_database_url: str) -> None:
    strict = create_app(
        Settings(database_url=None, app_env="test", allowed_hosts=("testserver",)),
        database_url=test_database_url,
    )
    client = TestClient(strict)

    response = client.get("/results", headers={"Host": "evil.example.com"}, follow_redirects=False)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Host not allowed"}
    assert "location" not in response.headers


# R-081: the old partial link still redirects byte for byte, then to `/`.


def test_old_partial_link_without_query_takes_two_relative_hops_to_landing(
    client: TestClient,
) -> None:
    first = client.get("/results/partials/content", follow_redirects=False)

    assert first.status_code == 302
    assert first.headers["location"] == "/results"

    response = client.get("/results/partials/content")

    assert response.status_code == 200
    assert [r.status_code for r in response.history] == [302, 302]
    assert [r.headers["location"] for r in response.history] == ["/results", "/"]
    assert str(response.url) == "http://testserver/"
