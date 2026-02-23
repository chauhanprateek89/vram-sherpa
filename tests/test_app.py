from __future__ import annotations

from sqlalchemy.orm import Session
from starlette.requests import Request

from vramsherpa.main import app, healthz, home, how_it_works, model_detail, results


def _request(path: str) -> Request:
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


def _response_text(response) -> str:
    return response.body.decode("utf-8")


def test_home_returns_200(seeded_session: Session) -> None:
    response = home(_request("/"), session=seeded_session)
    assert response.status_code == 200
    assert "VRAM Sherpa" in _response_text(response)


def test_results_with_manual_vram_returns_200_and_rows(seeded_session: Session) -> None:
    response = results(
        _request("/results"),
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
    assert "Required VRAM (GB)" in response_text
    assert "Available VRAM (GB)" in response_text
    assert "Margin (GB)" in response_text


def test_results_filters_change_output_deterministically(seeded_session: Session) -> None:
    llama_response = results(
        _request("/results"),
        vram_gb=24,
        context_tokens=2048,
        family=["Llama"],
        quant_bucket=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
        session=seeded_session,
    )
    qwen_response = results(
        _request("/results"),
        vram_gb=24,
        context_tokens=2048,
        family=["Qwen"],
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


def test_model_detail_returns_200(seeded_session: Session) -> None:
    response = model_detail(
        _request("/models/1"),
        model_id=1,
        vram_gb=16,
        context_tokens=2048,
        session=seeded_session,
    )
    assert response.status_code == 200
    assert "Llama 3.1 8B Instruct" in _response_text(response)


def test_how_it_works_returns_200(seeded_session: Session) -> None:
    response = how_it_works(_request("/how-it-works"), session=seeded_session)
    assert response.status_code == 200
    assert "not a benchmark" in _response_text(response)


def test_healthz_returns_json() -> None:
    assert healthz() == {"status": "ok"}
