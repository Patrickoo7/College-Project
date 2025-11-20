"""JWT Authentication module for API security.

This module provides JWT-based authentication for the API:
- Token generation and validation
- Password hashing and verification
- Authentication dependencies for protected endpoints
- All settings from config.api (enable_auth, secret_key, access_token_expire_minutes)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from ..config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Load configuration
config = get_config()

# JWT settings from config
SECRET_KEY = config.api.secret_key or "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = config.api.access_token_expire_minutes

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer authentication scheme
security = HTTPBearer(auto_error=False)


class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """JWT token payload data."""
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# In-memory user database (replace with actual database in production)
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to compare against

    Returns:
        True if passwords match, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a plain password.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database by username.

    Args:
        username: Username to look up

    Returns:
        User object if found, None otherwise
    """
    if username in FAKE_USERS_DB:
        user_dict = FAKE_USERS_DB[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password.

    Args:
        username: Username
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta. If None, uses config default.

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    logger.debug(f"Created access token for: {data.get('sub', 'unknown')}")

    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT access token.

    Args:
        token: JWT token to decode

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            return None

        token_data = TokenData(username=username, scopes=payload.get("scopes", []))
        return token_data

    except JWTError as e:
        logger.warning(f"JWT decode error: {str(e)}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user from JWT token.

    This is a FastAPI dependency that validates the JWT token and returns the user.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        User object if authenticated, None if authentication disabled

    Raises:
        HTTPException: If authentication is enabled but token is invalid
    """
    # If authentication is disabled in config, allow all requests
    if not config.api.enable_auth:
        logger.debug("Authentication is disabled in config")
        return None

    # Authentication is enabled, validate token
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    token_data = decode_access_token(token)

    if token_data is None or token_data.username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user(username=token_data.username)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    logger.debug(f"Authenticated user: {user.username}")

    return User(**user.dict())


async def get_current_active_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> Optional[User]:
    """Get current active user (not disabled).

    This is a convenience dependency that ensures the user is active.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User object if active, None if authentication disabled

    Raises:
        HTTPException: If user is disabled
    """
    if current_user is None:
        # Authentication disabled
        return None

    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user


def create_token_for_user(user: UserInDB) -> Token:
    """Create a JWT token for a user.

    Args:
        user: User to create token for

    Returns:
        Token response with access_token and metadata
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = create_access_token(
        data={"sub": user.username, "scopes": []},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    )


# Optional: Helper function to generate hashed password for new users
def generate_password_hash(password: str) -> str:
    """Generate a hashed password (for development/testing).

    Example usage:
        hashed = generate_password_hash("mypassword")
        print(f"Hashed password: {hashed}")

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return get_password_hash(password)
