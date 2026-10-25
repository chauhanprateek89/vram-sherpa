from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main
from vramsherpa.config import Settings
from vramsherpa.main import create_app

QUERY = "vram_gb=24&context_tokens=4096"
HX = {"HX-Request": "true"}

# The nine R-080 check URLs, with the extra headers each one is requested with.
R080_CHECK_URLS = [
    ("/", {}),
    (f"/results?{QUERY}", {}),
    (f"/results?{QUERY}", HX),
    (f"/results/partials/content?{QUERY}", HX),
    (f"/results/partials/summary?{QUERY}", {}),
    (f"/results/partials/list?{QUERY}", {}),
    (f"/results/why/variant_model_qwen25_7b_instruct_q4?{QUERY}", {}),
    (f"/models/model_qwen25_7b_instruct?{QUERY}", {}),
    ("/how-it-works", {}),
]


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def _vary_names(response) -> list[str]:
    names: list[str] = []
    for value in response.headers.get_list("vary"):
        names.extend(part.strip().lower() for part in value.split(",") if part.strip())
    return names


def _max_age_values(cache_control: str) -> list[int]:
    values: list[int] = []
    for directive in cache_control.split(","):
        name, _, value = directive.strip().partition("=")
        if name.strip().lower() in {"max-age", "s-maxage"}:
            values.append(int(value.strip().strip('"')))
    return values


@pytest.mark.parametrize(("url", "headers"), R080_CHECK_URLS)
def test_html_check_urls_send_one_no_cache_header(
    client: TestClient, url: str, headers: dict[str, str]
) -> None:
    response = client.get(url, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    cache_control_values = response.headers.get_list("cache-control")
    assert len(cache_control_values) == 1
    cache_control = cache_control_values[0]
    assert "no-cache" in cache_control
    assert "immutable" not in cache_control
    assert all(value <= 0 for value in _max_age_values(cache_control))


@pytest.mark.parametrize(("url", "headers"), R080_CHECK_URLS)
def test_html_check_urls_vary_on_both_htmx_headers(
    client: TestClient, url: str, headers: dict[str, str]
) -> None:
    response = client.get(url, headers=headers)

    names = _vary_names(response)
    assert "hx-request" in names
    assert "hx-history-restore-request" in names
    assert len(names) == len(set(names))


def test_healthz_has_no_cache_control(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "cache-control" not in response.headers


def test_static_css_has_no_cache_control(client: TestClient) -> None:
    response = client.get("/static/css/styles.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "cache-control" not in response.headers


def test_unknown_path_404_has_no_cache_control(client: TestClient) -> None:
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "cache-control" not in response.headers


@pytest.mark.parametrize("path", ["/", "/healthz", f"/results?{QUERY}"])
def test_unknown_host_still_rejected_without_cache_control(client: TestClient, path: str) -> None:
    response = client.get(path, headers={"Host": "evil.example.com"})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Host not allowed"}
    assert "cache-control" not in response.headers


def test_handler_cache_control_is_kept_and_vary_is_appended(test_database_url: str) -> None:
    app = create_app(
        Settings(database_url=None, app_env="test", allowed_hosts=("testserver",)),
        database_url=test_database_url,
    )

    def custom_cache() -> HTMLResponse:
        return HTMLResponse(
            "<p>custom</p>",
            headers={"Cache-Control": "no-store", "Vary": "Accept-Encoding"},
        )

    app.add_api_route("/test-only/custom-cache", custom_cache, methods=["GET"])

    response = TestClient(app).get("/test-only/custom-cache")

    assert response.status_code == 200
    assert response.headers.get_list("cache-control") == ["no-store"]
    names = _vary_names(response)
    assert names[0] == "accept-encoding"
    assert "hx-request" in names
    assert "hx-history-restore-request" in names
    assert len(names) == len(set(names))


def test_existing_htmx_vary_names_are_not_duplicated(test_database_url: str) -> None:
    app = create_app(
        Settings(database_url=None, app_env="test", allowed_hosts=("testserver",)),
        database_url=test_database_url,
    )

    def already_varies() -> HTMLResponse:
        return HTMLResponse("<p>vary</p>", headers={"Vary": "hx-request"})

    app.add_api_route("/test-only/already-varies", already_varies, methods=["GET"])

    response = TestClient(app).get("/test-only/already-varies")

    assert response.headers.get_list("cache-control") == ["no-cache"]
    assert sorted(_vary_names(response)) == ["hx-history-restore-request", "hx-request"]


def test_middleware_keys_on_content_type_and_sits_beside_host_guard() -> None:
    source = inspect.getsource(main.html_cache_control)
    assert "content-type" in source
    # The middleware decides on the response alone; it never looks at the request path.
    assert "request.url" not in source
    assert "scope[" not in source

    create_app_source = inspect.getsource(main.create_app)
    assert 'app.middleware("http")(html_cache_control)' in create_app_source
    assert 'app.middleware("http")(host_guard)' in create_app_source
