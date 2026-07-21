import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api_mcp import app


client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")

    assert response.json() == {"status": "ok"}


def test_root_returns_200():
    response = client.get("/")

    assert response.status_code == 200


def test_root_contains_basic_html_content():
    response = client.get("/")

    assert "<html" in response.text.lower()
    assert "rag chat" in response.text.lower()