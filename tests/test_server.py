"""Tests for the FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient

from Kmcalgo.astrum_env.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["env"] == "hypernoa_astrum"


class TestRootEndpoint:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "env" in data


class TestResetEndpoint:
    def test_reset(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "stakeholders" in data
        assert "resources" in data
        assert data["step_count"] == 0

    def test_reset_with_seed(self, client):
        resp = client.post("/reset", json={"seed": 42})
        assert resp.status_code == 200


class TestStepEndpoint:
    def test_step(self, client):
        client.post("/reset")
        resp = client.post("/step", json={
            "action_type": "noop",
            "params": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["step_count"] == 1
        assert "reward" in data
        assert "reward_breakdown" in data

    def test_step_allocate(self, client):
        client.post("/reset")
        resp = client.post("/step", json={
            "action_type": "allocate_resources",
            "params": {"stakeholder": "workers", "amount": 10, "resource": "budget"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert any("allocated" in a for a in data.get("alerts", []))

    def test_full_episode(self, client):
        client.post("/reset")
        for _ in range(32):
            resp = client.post("/step", json={
                "action_type": "noop",
                "params": {},
            })
        data = resp.json()
        assert data["done"] is True
