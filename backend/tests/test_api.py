import pytest
from fastapi.testclient import TestClient

import app.main as main

# No `with TestClient(...)`: skips the lifespan, so tests don't load the real index/model.
client = TestClient(main.app)


class _EmptyIndex:
    records = []

    def search(self, question, k_text, k_figures):
        return [], []


@pytest.fixture(autouse=True)
def stub_index(monkeypatch):
    monkeypatch.setitem(main._state, "index", _EmptyIndex())
    monkeypatch.setattr(main, "_providers", {})


@pytest.mark.parametrize("path", [
    "/api/figures/..%2F..%2F.env/fig1_1.png",
    "/api/figures/2609.00001v1/paper.json",
    "/api/figures/2609.00001v1/embeddings.npy",
    "/api/figures/2609.00001v1/fig1_1.png",  # well-formed but doesn't exist
])
def test_figure_route_only_serves_existing_figure_pngs(path):
    assert client.get(path).status_code == 404


def test_ask_reports_missing_key_clearly(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    response = client.post("/api/ask", json={"question": "What is sigma_8?", "provider": "deepseek"})
    assert response.status_code == 400
    assert "DEEPSEEK_API_KEY" in response.json()["detail"]


def test_ask_validates_input():
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "provider": "gpt"}).status_code == 422
