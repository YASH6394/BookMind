from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_reports_default_pdf_index() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["indexed_chunks"] > 0


def test_query_returns_answer_and_sources() -> None:
    response = client.post(
        "/query",
        json={"question": "Who is Heathcliff?", "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["sources"]) > 0
    assert all(source["page"] >= 1 for source in body["sources"])


def test_unknown_terms_return_a_clear_response() -> None:
    response = client.post(
        "/query",
        json={"question": "xyzzynonsensequantumtoken", "top_k": 5},
    )
    assert response.status_code == 200
    assert "could not find" in response.json()["answer"].lower()
