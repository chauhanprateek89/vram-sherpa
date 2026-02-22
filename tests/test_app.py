from __future__ import annotations

from fastapi.testclient import TestClient


def test_home_returns_200(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "VRAM Sherpa" in response.text


def test_results_with_manual_vram_returns_200_and_rows(client: TestClient) -> None:
    response = client.get("/results", params={"vram_gb": 16, "context_tokens": 2048})
    assert response.status_code == 200
    assert "Required VRAM (GB)" in response.text
    assert "Available VRAM (GB)" in response.text
    assert "Margin (GB)" in response.text


def test_results_filters_change_output_deterministically(client: TestClient) -> None:
    llama_response = client.get(
        "/results",
        params=[("vram_gb", "24"), ("context_tokens", "2048"), ("family", "Llama")],
    )
    qwen_response = client.get(
        "/results",
        params=[("vram_gb", "24"), ("context_tokens", "2048"), ("family", "Qwen")],
    )

    assert llama_response.status_code == 200
    assert qwen_response.status_code == 200

    assert "Llama 3.1 8B Instruct" in llama_response.text
    assert "Qwen2.5 14B Instruct" not in llama_response.text

    assert "Qwen2.5 14B Instruct" in qwen_response.text
    assert "Llama 3.1 8B Instruct" not in qwen_response.text


def test_model_detail_returns_200(client: TestClient) -> None:
    response = client.get("/models/1", params={"vram_gb": 16, "context_tokens": 2048})
    assert response.status_code == 200
    assert "Llama 3.1 8B Instruct" in response.text


def test_how_it_works_returns_200(client: TestClient) -> None:
    response = client.get("/how-it-works")
    assert response.status_code == 200
    assert "not a benchmark" in response.text


def test_healthz_returns_json(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
