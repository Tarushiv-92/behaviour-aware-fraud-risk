from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "expected_features" in r.json()


def test_score_stub():
    r = client.post(
        "/score",
        json={
            "customer_id": "C0001",
            "amount": 8000,
            "merchant": "Unknown",
            "beneficiary": "wallet",
            "timestamp": "2026-06-01T03:12:00",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "risk_score" in body
    assert "risk_band" in body
    assert "reason_codes" in body
