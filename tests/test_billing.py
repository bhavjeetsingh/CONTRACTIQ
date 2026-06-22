"""
Unit tests for Razorpay billing endpoints in api/main.py.
"""
import uuid
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_create_order_unauthorized():
    # Attempting to create order without bearer token should fail
    res = client.post("/billing/order", json={"tier": "premium"})
    assert res.status_code in [401, 403]


def test_verify_payment_unauthorized():
    # Attempting to verify signature without bearer token should fail
    res = client.post("/billing/verify", json={
        "razorpay_order_id": "order_mock_123",
        "razorpay_payment_id": "pay_123",
        "razorpay_signature": "sig_123"
    })
    assert res.status_code in [401, 403]


def test_order_creation_and_upgrade_flow():
    # Register and login a new user
    email = f"billing_{uuid.uuid4().hex[:8]}@example.com"
    password = "billingpassword123"
    
    # Register
    client.post("/auth/register", json={"email": email, "password": password})
    
    # Login
    login_res = client.post("/auth/login", json={"email": email, "password": password})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Create Order
    order_res = client.post("/billing/order", json={"tier": "premium"}, headers=headers)
    assert order_res.status_code == 200
    order_data = order_res.json()
    assert "id" in order_data
    assert order_data["amount"] == 49900
    assert "razorpay_key_id" in order_data
    
    # 2. Verify Payment (using mock verification path)
    verify_res = client.post("/billing/verify", json={
        "razorpay_order_id": order_data["id"],
        "razorpay_payment_id": "pay_mock_999888",
        "razorpay_signature": "sig_mock_777666"
    }, headers=headers)
    
    assert verify_res.status_code == 200
    verify_data = verify_res.json()
    assert verify_data["status"] == "verified"
    assert verify_data["tier"] == "premium"
