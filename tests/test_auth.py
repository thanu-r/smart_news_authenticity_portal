import pytest
from backend.app.auth.security import hash_password, verify_password, create_access_token, decode_access_token
from backend.app.schemas.user_schemas import UserRegister, UserLogin

def test_password_hashing_and_verification():
    """Verify that bcrypt hashes correctly and verifies plain passwords."""
    raw_pw = "SecureResearchPassword123!"
    hashed = hash_password(raw_pw)
    assert hashed != raw_pw
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert verify_password(raw_pw, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False

def test_jwt_token_lifecycle():
    """Verify token encoding and decoding."""
    payload = {"sub": "65f1234567890abcdef12345", "role": "ADMIN"}
    token = create_access_token(payload)
    assert isinstance(token, str)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == payload["sub"]
    assert decoded["role"] == payload["role"]
    assert "exp" in decoded

def test_invalid_token_decoding():
    """Ensure tampered tokens are rejected."""
    tampered = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalidpayload.invalidsignature"
    assert decode_access_token(tampered) is None

def test_user_schemas_validation():
    """Verify pydantic model validations for user registration and login."""
    reg = UserRegister(
        email="researcher@university.edu",
        username="lead_researcher",
        password="ValidPassword123"
    )
    assert reg.email == "researcher@university.edu"
    assert reg.username == "lead_researcher"

    login = UserLogin(
        username_or_email="lead_researcher",
        password="ValidPassword123"
    )
    assert login.username_or_email == "lead_researcher"

def test_api_health_endpoint():
    """Verify health endpoint confirms database and service status."""
    import uuid
    from starlette.testclient import TestClient
    from backend.app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database_connected"] is True

def test_auth_end_to_end_flow():
    """Verify registration, duplicate rejection, login, and authenticated /me retrieval."""
    import uuid
    from starlette.testclient import TestClient
    from backend.app.main import app

    uid = uuid.uuid4().hex[:8]
    email = f"user_{uid}@newsverify.org"
    username = f"analyst_{uid}"
    password = f"SecurePw99!_{uid}"

    with TestClient(app) as client:
        # 1. Register new user
        reg_payload = {
            "email": email,
            "username": username,
            "password": password,
            "full_name": "Senior Fact Checker"
        }
        reg_resp = client.post("/api/auth/register", json=reg_payload)
        assert reg_resp.status_code == 201
        reg_data = reg_resp.json()
        assert "access_token" in reg_data
        token = reg_data["access_token"]
        assert reg_data["user"]["email"] == email
        assert reg_data["user"]["username"] == username

        # 2. Duplicate registration should fail
        dup_resp = client.post("/api/auth/register", json=reg_payload)
        assert dup_resp.status_code == 400

        # 3. Login with credentials
        login_resp = client.post("/api/auth/login", json={
            "username_or_email": email,
            "password": password
        })
        assert login_resp.status_code == 200
        login_data = login_resp.json()
        assert login_data["access_token"] is not None

        # 4. Login with invalid password fails
        bad_login = client.post("/api/auth/login", json={
            "username_or_email": email,
            "password": "WrongPassword123"
        })
        assert bad_login.status_code == 401

        # 5. Access /me with valid token
        me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        me_data = me_resp.json()
        assert me_data["email"] == email
        assert me_data["username"] == username

        # 6. Access /me without token fails
        anon_resp = client.get("/api/auth/me")
        assert anon_resp.status_code == 401

