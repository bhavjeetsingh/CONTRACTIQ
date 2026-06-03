# tests/test_unit_cases.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "ContractIQ" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "document-portal"


def test_supported_formats():
    response = client.get("/supported-formats")
    assert response.status_code == 200
    data = response.json()
    assert "supported_extensions" in data
    assert ".pdf" in data["supported_extensions"]


def test_register_and_login():
    import uuid
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"

    # Register
    res = client.post("/auth/register", json={"email": email, "password": password})
    assert res.status_code == 200
    assert "registered" in res.json()["message"].lower()

    # Login
    res = client.post("/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_register_duplicate_email():
    import uuid
    email = f"dup_{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"

    client.post("/auth/register", json={"email": email, "password": password})
    res = client.post("/auth/register", json={"email": email, "password": password})
    assert res.status_code == 400


def test_login_wrong_password():
    import uuid
    email = f"wrong_{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "correctpass123"})
    res = client.post("/auth/login", json={"email": email, "password": "wrongpassword"})
    assert res.status_code == 401


def test_register_weak_password():
    res = client.post("/auth/register", json={"email": "weak@test.com", "password": "123"})
    assert res.status_code == 400


def test_protected_endpoint_without_token():
    res = client.get("/sessions")
    assert res.status_code in [401, 403]


def test_unsupported_file_type():
    import uuid
    email = f"file_{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "testpassword123"})
    login = client.post("/auth/login", json={"email": email, "password": "testpassword123"})
    token = login.json()["access_token"]

    res = client.post(
        "/analyze",
        files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400
