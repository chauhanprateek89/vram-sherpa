from __future__ import annotations

import re

import pytest
from sqlalchemy.orm import Session
from starlette.requests import Request

from vramsherpa.config import Settings
from vramsherpa.main import (
    create_app,
    healthz,
    home,
    how_it_works,
    model_detail,
    results,
    results_partial_content,
    results_partial_list,
    results_partial_summary,
    variant_breakdown,
)


def _request(app, path: str, *, hx: bool = False) -> Request:
    headers = [(b"host", b"testserver")]
    if hx:
        headers.append((b"hx-request", b"true"))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "app": app,
            "router": app.router,
        }
    )


def _response_text(response) -> str:
    return response.body.decode("utf-8")


def test_home_returns_200(app, seeded_session: Session) -> None:
    response = home(_request(app, "/"), session=seeded_session)
    assert response.status_code == 200
    assert "VRAM Sherpa" in _response_text(response)


def test_results_with_manual_vram_returns_200_and_rows(app, seeded_session: Session) -> None:
    response = results(
        _request(app, "/results"),
        vram_gb=16,
        context_tokens=2048,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    assert response.status_code == 200
    response_text = _response_text(response)
    assert 'id="summary-banner"' in response_text
    assert "result-card" in response_text
    assert "Hardware:" in response_text
    assert "Context:" in response_text


def test_results_accepts_blank_numeric_query_values(app, seeded_session: Session) -> None:
    response = results(
        _request(app, "/results"),
        gpu_id="",
        gpu_search="",
        vram_gb="8",
        context_tokens=2048,
        family=[],
        quant_bucket=[],
        min_params_b="",
        max_params_b="",
        recommended_only=False,
        session=seeded_session,
    )
    assert response.status_code == 200
    assert 'id="summary-banner"' in _response_text(response)


def test_results_invalid_values_render_inline_errors(app, seeded_session: Session) -> None:
    response = results(
        _request(app, "/results"),
        gpu_id="",
        gpu_search="",
        vram_gb="abc",
        context_tokens=2048,
        family=[],
        quant_bucket=[],
        min_params_b="20",
        max_params_b="5",
        recommended_only=False,
        session=seeded_session,
    )
    assert response.status_code == 200
    response_text = _response_text(response)
    assert "Invalid value for vram_gb." in response_text
    assert "min_params_b must be less than or equal to max_params_b." in response_text


def test_results_gpu_search_does_not_auto_pick_ambiguous_match(
    app, seeded_session: Session
) -> None:
    response = results(
        _request(app, "/results"),
        gpu_id="",
        gpu_search="RTX",
        vram_gb="",
        context_tokens=2048,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    assert response.status_code == 200
    response_text = _response_text(response)
    assert "matched multiple entries" in response_text
    hidden_gpu_match = re.search(
        r'<input id="gpu_id" name="gpu_id" type="hidden" value="([^"]*)"',
        response_text,
    )
    assert hidden_gpu_match is not None
    assert hidden_gpu_match.group(1) == ""


def test_results_gpu_search_shows_not_found_feedback(app, seeded_session: Session) -> None:
    response = results(
        _request(app, "/results"),
        gpu_id="",
        gpu_search="notarealgpu123",
        vram_gb="",
        context_tokens=2048,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    assert response.status_code == 200
    response_text = _response_text(response)
    assert "notarealgpu123" in response_text
    assert "was not found" in response_text


def test_results_gpu_search_uses_vram_hint_from_text(app, seeded_session: Session) -> None:
    response = results(
        _request(app, "/results"),
        gpu_id="",
        gpu_search="RTX 3060 12GB",
        vram_gb="",
        context_tokens=2048,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    assert response.status_code == 200
    response_text = _response_text(response)
    assert "Using 12 GB inferred from your input." in response_text
    assert ">12.00 GB<" in response_text


def test_variant_breakdown_returns_html_partial(app, seeded_session: Session) -> None:
    response = variant_breakdown(
        _request(app, "/results/why/variant_model_llama_31_8b_instruct_q4"),
        variant_id="variant_model_llama_31_8b_instruct_q4",
        vram_gb=16,
        context_tokens=2048,
        session=seeded_session,
    )

    assert response.status_code == 200
    response_text = _response_text(response)
    assert "weights_gb" in response_text
    assert "kv_cache_gb" in response_text
    assert "runtime_overhead_gb" in response_text
    assert "reserve_gb" in response_text


def test_results_filters_change_output_deterministically(app, seeded_session: Session) -> None:
    llama_response = results(
        _request(app, "/results"),
        vram_gb=24,
        context_tokens=2048,
        family=["llama"],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    qwen_response = results(
        _request(app, "/results"),
        vram_gb=24,
        context_tokens=2048,
        family=["qwen"],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )

    assert llama_response.status_code == 200
    assert qwen_response.status_code == 200

    llama_text = _response_text(llama_response)
    qwen_text = _response_text(qwen_response)

    assert "Llama 3.1 8B Instruct" in llama_text
    assert "Qwen2.5 14B Instruct" not in llama_text

    assert "Qwen2.5 14B Instruct" in qwen_text
    assert "Llama 3.1 8B Instruct" not in qwen_text


def test_results_partials_return_summary_and_results_sections(
    app, seeded_session: Session
) -> None:
    summary_response = results_partial_summary(
        _request(app, "/results/partials/summary", hx=True),
        vram_gb="16",
        context_tokens=4096,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    list_response = results_partial_list(
        _request(app, "/results/partials/list", hx=True),
        vram_gb="16",
        context_tokens=4096,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    content_response = results_partial_content(
        _request(app, "/results/partials/content", hx=True),
        vram_gb="16",
        context_tokens=4096,
        family=[],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )

    assert summary_response.status_code == 200
    assert list_response.status_code == 200
    assert content_response.status_code == 200
    assert 'id="summary-banner"' in _response_text(summary_response)
    assert 'id="results-cards"' in _response_text(list_response)
    content_text = _response_text(content_response)
    assert 'id="results-summary-shell"' in content_text
    assert 'id="results-list-shell"' in content_text


def test_model_detail_returns_200(app, seeded_session: Session) -> None:
    response = model_detail(
        _request(app, "/models/model_llama_31_8b_instruct"),
        model_id="model_llama_31_8b_instruct",
        vram_gb=16,
        context_tokens=2048,
        session=seeded_session,
    )
    assert response.status_code == 200
    assert "Llama 3.1 8B Instruct" in _response_text(response)


def test_model_detail_accepts_blank_manual_vram(app, seeded_session: Session) -> None:
    response = model_detail(
        _request(app, "/models/model_llama_31_8b_instruct"),
        model_id="model_llama_31_8b_instruct",
        gpu_id="gpu_nvidia_rtx_3090_24gb",
        vram_gb="",
        context_tokens=2048,
        session=seeded_session,
    )
    assert response.status_code == 200


def test_model_detail_prefers_gpu_selection_when_both_gpu_and_manual_vram_provided(
    app, seeded_session: Session
) -> None:
    response = model_detail(
        _request(app, "/models/model_llama_31_8b_instruct"),
        model_id="model_llama_31_8b_instruct",
        gpu_id="gpu_nvidia_rtx_3090_24gb",
        vram_gb="8",
        context_tokens=2048,
        session=seeded_session,
    )
    assert response.status_code == 200
    response_text = _response_text(response)
    assert 'option value="gpu_nvidia_rtx_3090_24gb" selected' in response_text
    assert ">24.00<" in response_text


def test_model_detail_invalid_manual_vram_renders_inline_error(
    app, seeded_session: Session
) -> None:
    response = model_detail(
        _request(app, "/models/model_llama_31_8b_instruct"),
        model_id="model_llama_31_8b_instruct",
        gpu_id="",
        vram_gb="abc",
        context_tokens=2048,
        session=seeded_session,
    )
    assert response.status_code == 200
    assert "Invalid value for vram_gb." in _response_text(response)


def test_how_it_works_returns_200(app, seeded_session: Session) -> None:
    response = how_it_works(_request(app, "/how-it-works"), session=seeded_session)
    assert response.status_code == 200
    assert "not a benchmark" in _response_text(response)


def test_healthz_returns_json() -> None:
    assert healthz() == {"status": "ok"}


def test_create_app_requires_database_url_outside_test_env() -> None:
    with pytest.raises(RuntimeError, match="APP_ENV is not 'test'"):
        create_app(
            Settings(
                database_url=None,
                app_env="dev",
                allowed_hosts=("testserver",),
            )
        )


def test_create_app_requires_database_url_for_test_env_when_not_explicit() -> None:
    with pytest.raises(RuntimeError, match="APP_ENV is 'test'"):
        create_app(
            Settings(
                database_url=None,
                app_env="test",
                allowed_hosts=("testserver",),
            )
        )


def test_create_app_accepts_explicit_database_url_for_test_env() -> None:
    app = create_app(
        Settings(
            database_url=None,
            app_env="test",
            allowed_hosts=("testserver",),
        ),
        database_url="sqlite+pysqlite:///:memory:",
    )
    assert app.title == "VRAM Sherpa"
