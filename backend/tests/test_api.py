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


class _FakeProvider:
    def answer(self, question, sources):
        return main.Answer("deepseek", "m", "ok", [], [], 1, 1, 0.1)


def test_ask_rate_limit_per_client_and_per_day(monkeypatch):
    monkeypatch.setattr(main, "_providers", {"deepseek": _FakeProvider(), "claude": _FakeProvider()})
    monkeypatch.setattr(main, "_limiter", main.AskLimiter(per_ip_per_hour=2, per_day=3))
    ask = lambda ip, provider="deepseek": client.post(  # noqa: E731
        "/api/ask", json={"question": "q", "provider": provider}, headers={"X-Forwarded-For": ip}
    ).status_code
    assert [ask("1.1.1.1"), ask("1.1.1.1"), ask("1.1.1.1")] == [200, 200, 429]
    assert ask("2.2.2.2") == 200  # different client, but that's the 3rd of 3 daily calls
    assert ask("3.3.3.3") == 429
    assert client.post("/api/search", json={"question": "q"}).status_code == 200  # search is never limited


def test_ask_both_counts_two_calls_against_daily_budget(monkeypatch):
    monkeypatch.setattr(main, "_providers", {"deepseek": _FakeProvider(), "claude": _FakeProvider()})
    monkeypatch.setattr(main, "_limiter", main.AskLimiter(per_ip_per_hour=0, per_day=3))
    both = {"question": "q", "provider": "both"}
    assert client.post("/api/ask", json=both).status_code == 200
    assert client.post("/api/ask", json=both).status_code == 429


def test_reload_index_requires_admin_token_when_set(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/api/reload-index").status_code == 403
    assert client.post("/api/reload-index", headers={"X-Admin-Token": "wrong"}).status_code == 403
