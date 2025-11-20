# JWT Authentication Guide

This guide explains how to use and configure the JWT authentication system in the Heart Disease Prediction API.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Authentication Flow](#authentication-flow)
- [Using the API](#using-the-api)
- [Protecting Routes](#protecting-routes)
- [Adding New Users](#adding-new-users)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The API includes a complete JWT (JSON Web Token) based authentication system:

- **Config-Driven**: All settings from `config.api` (enable_auth, secret_key, token expiration)
- **Optional**: Authentication can be enabled/disabled via configuration
- **Standard JWT**: Uses industry-standard JWT tokens with HS256 algorithm
- **FastAPI Integration**: Seamless integration with FastAPI dependencies
- **Password Hashing**: Secure bcrypt password hashing
- **Middleware Support**: Optional authentication middleware for global enforcement

---

## Configuration

### Enable/Disable Authentication

Authentication is controlled by the `enable_auth` flag in your configuration:

```yaml
# configs/app_config.yaml
api:
  enable_auth: false  # Set to true to enable authentication
  secret_key: null    # Set a secret key for JWT signing
  access_token_expire_minutes: 30
```

### Environment-Specific Configuration

**Development (authentication disabled):**
```yaml
# configs/development_config.yaml
api:
  enable_auth: false
```

**Production (authentication enabled):**
```yaml
# configs/production_config.yaml
api:
  enable_auth: true
  secret_key: "${APP_API__SECRET_KEY}"  # Load from environment variable
  access_token_expire_minutes: 30
```

### Environment Variables

You can override configuration using environment variables:

```bash
# Enable authentication
export APP_API__ENABLE_AUTH=true

# Set secret key (IMPORTANT: Use a strong secret in production!)
export APP_API__SECRET_KEY="your-super-secret-key-here-change-this"

# Set token expiration (minutes)
export APP_API__ACCESS_TOKEN_EXPIRE_MINUTES=60
```

**Generate a strong secret key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Authentication Flow

### 1. Login to Get Token

**Endpoint:** `POST /api/v1/auth/login`

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "username": "admin",
    "email": "admin@example.com",
    "full_name": "Admin User",
    "disabled": false
  }
}
```

### 2. Use Token for Authenticated Requests

Include the token in the `Authorization` header:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": 1, "cp": 0, ...}'
```

### 3. Token Validation

**Endpoint:** `POST /api/v1/auth/validate`

```bash
curl -X POST "http://localhost:8000/api/v1/auth/validate" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

**Response:**
```json
{
  "valid": true,
  "username": "admin"
}
```

---

## Using the API

### Python Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# 1. Login
login_response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "admin", "password": "secret"}
)

token_data = login_response.json()
access_token = token_data["access_token"]

# 2. Use token for authenticated requests
headers = {"Authorization": f"Bearer {access_token}"}

# Make prediction
prediction_response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json={
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
)

result = prediction_response.json()
print(f"Prediction: {result['prediction_label']}")
```

### JavaScript/TypeScript Example

```typescript
// 1. Login
const loginResponse = await fetch('http://localhost:8000/api/v1/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
  },
  body: 'username=admin&password=secret'
});

const { access_token } = await loginResponse.json();

// 2. Use token for authenticated requests
const predictionResponse = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    age: 55,
    sex: 1,
    cp: 0,
    // ... other patient data
  })
});

const result = await predictionResponse.json();
console.log(`Prediction: ${result.prediction_label}`);
```

---

## Protecting Routes

### Method 1: Using Depends (Recommended)

Protect individual routes by adding the `get_current_active_user` dependency:

```python
from fastapi import APIRouter, Depends
from .auth import get_current_active_user, User

router = APIRouter()

@router.post("/protected-endpoint")
async def protected_route(
    current_user: User = Depends(get_current_active_user)
):
    """
    This endpoint requires authentication if config.api.enable_auth is True.

    If authentication is disabled, current_user will be None.
    If authentication is enabled, current_user will contain the authenticated user.
    """
    if current_user:
        # User is authenticated
        return {"message": f"Hello, {current_user.username}!"}
    else:
        # Authentication is disabled
        return {"message": "Hello, anonymous user!"}
```

### Method 2: Using Middleware (Global)

Enable authentication middleware in `src/api/app.py`:

```python
# Uncomment these lines in app.py
from .middleware import AuthenticationMiddleware
app.add_middleware(AuthenticationMiddleware)
```

This will enforce authentication globally for all routes except public routes defined in the middleware.

---

## Adding New Users

### Current Implementation

The current implementation uses an in-memory user database for demonstration purposes:

```python
# src/api/auth.py
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "hashed_password": "$2b$12$...",  # "secret"
        "disabled": False,
    }
}
```

### Adding a New User

1. **Generate hashed password:**

```python
from src.api.auth import generate_password_hash

hashed = generate_password_hash("newpassword")
print(f"Hashed password: {hashed}")
```

2. **Add to FAKE_USERS_DB:**

```python
FAKE_USERS_DB["newuser"] = {
    "username": "newuser",
    "email": "newuser@example.com",
    "full_name": "New User",
    "hashed_password": hashed,
    "disabled": False,
}
```

### Production Implementation

For production, replace the in-memory database with a real database:

1. **Create User Model:**
```python
from sqlalchemy import Column, String, Boolean
from database import Base

class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True)
    email = Column(String, unique=True)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
```

2. **Update `get_user()` in auth.py:**
```python
def get_user(username: str) -> Optional[UserInDB]:
    # Query database instead of in-memory dict
    user = db.query(User).filter(User.username == username).first()
    if user:
        return UserInDB(**user.__dict__)
    return None
```

---

## Security Best Practices

### 1. Secret Key Management

**❌ Don't:**
```yaml
api:
  secret_key: "hardcoded-secret-key"  # Bad!
```

**✅ Do:**
```bash
# .env file (never commit to git!)
APP_API__SECRET_KEY="generated-secure-key-here"
```

```yaml
# config file
api:
  secret_key: "${APP_API__SECRET_KEY}"
```

### 2. Token Expiration

Set appropriate token expiration based on security requirements:

```yaml
# Short-lived tokens (more secure)
api:
  access_token_expire_minutes: 15

# Longer-lived tokens (more convenient)
api:
  access_token_expire_minutes: 60
```

### 3. HTTPS Only

Always use HTTPS in production:

```nginx
# nginx configuration
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### 4. Rate Limiting

Enable rate limiting to prevent brute-force attacks:

```python
# In app.py
from .middleware import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
```

### 5. Password Requirements

Enforce strong password requirements:

```python
import re

def validate_password(password: str) -> bool:
    """
    Validate password strength:
    - At least 8 characters
    - Contains uppercase and lowercase
    - Contains numbers
    - Contains special characters
    """
    if len(password) < 8:
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True
```

---

## Troubleshooting

### Issue: "Not authenticated" error

**Cause:** Missing or invalid token

**Solution:**
```bash
# Check if authentication is enabled
curl http://localhost:8000/api/v1/auth/status

# Login to get a valid token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=secret"

# Use the token in subsequent requests
curl -H "Authorization: Bearer YOUR_TOKEN" ...
```

### Issue: "Authentication is disabled" message

**Cause:** `enable_auth` is set to `false` in configuration

**Solution:**
```bash
# Enable via environment variable
export APP_API__ENABLE_AUTH=true

# Or update config file
# configs/app_config.yaml
api:
  enable_auth: true
```

### Issue: Token expired

**Cause:** Token has exceeded `access_token_expire_minutes`

**Solution:**
```bash
# Request a new token by logging in again
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=secret"
```

### Issue: "Incorrect username or password"

**Cause:** Invalid credentials

**Solution:**
- Verify username and password
- Default credentials: username=`admin`, password=`secret`
- Check user exists in `FAKE_USERS_DB` (or database in production)

### Issue: Token validation fails

**Cause:** Secret key mismatch or token corruption

**Solution:**
```bash
# Ensure consistent SECRET_KEY across all instances
export APP_API__SECRET_KEY="same-key-everywhere"

# Request a new token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=secret"
```

---

## API Endpoints Summary

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/auth/login` | POST | No | Login and get JWT token |
| `/auth/me` | GET | Yes | Get current user info |
| `/auth/validate` | POST | Yes | Validate JWT token |
| `/auth/status` | GET | No | Check if auth is enabled |
| `/predict` | POST | Optional* | Make prediction |
| `/predict/batch` | POST | Optional* | Batch predictions |
| `/models` | GET | Optional* | List available models |
| `/health` | GET | No | Health check |

*Required if `enable_auth: true` in configuration

---

## Testing Authentication

### Test Script

```python
#!/usr/bin/env python3
"""Test authentication system."""

import requests

BASE_URL = "http://localhost:8000/api/v1"

def test_auth():
    # 1. Check auth status
    status = requests.get(f"{BASE_URL}/auth/status").json()
    print(f"Auth enabled: {status['enabled']}")

    if not status['enabled']:
        print("Authentication is disabled, skipping tests")
        return

    # 2. Test invalid login
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": "admin", "password": "wrong"}
    )
    assert response.status_code == 401, "Invalid login should fail"
    print("✓ Invalid login rejected")

    # 3. Test valid login
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": "admin", "password": "secret"}
    )
    assert response.status_code == 200, "Valid login should succeed"
    token = response.json()["access_token"]
    print("✓ Valid login succeeded")

    # 4. Test protected endpoint without token
    response = requests.get(f"{BASE_URL}/auth/me")
    assert response.status_code == 401, "Protected route without token should fail"
    print("✓ Protected endpoint requires token")

    # 5. Test protected endpoint with token
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    assert response.status_code == 200, "Protected route with token should succeed"
    print("✓ Protected endpoint accepts valid token")

    print("\n✅ All authentication tests passed!")

if __name__ == "__main__":
    test_auth()
```

---

## Additional Resources

- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT.io](https://jwt.io/) - JWT debugger and documentation
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

---

**Last Updated:** 2025-11-20
