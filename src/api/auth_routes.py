"""Authentication routes for JWT token management.

This module provides endpoints for:
- User login (token generation)
- Token validation
- User information retrieval
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from .auth import (
    authenticate_user,
    create_token_for_user,
    get_current_active_user,
    Token,
    User,
)
from ..config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

# Load configuration
config = get_config()


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str
    expires_in: int
    user: User


@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    User login endpoint to obtain JWT access token.

    This endpoint authenticates users and returns a JWT token that can be used
    for subsequent API requests.

    Args:
        form_data: OAuth2 password form with username and password

    Returns:
        JWT token and user information

    Raises:
        HTTPException: If authentication fails

    Example:
        POST /api/v1/auth/login
        {
            "username": "admin",
            "password": "secret"
        }

        Response:
        {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
            "token_type": "bearer",
            "expires_in": 1800,
            "user": {
                "username": "admin",
                "email": "admin@example.com",
                "full_name": "Admin User",
                "disabled": false
            }
        }
    """
    if not config.api.enable_auth:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication is disabled in configuration"
        )

    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_token_for_user(user)

    logger.info(f"User logged in successfully: {user.username}")

    return LoginResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        expires_in=token.expires_in,
        user=User(**user.dict())
    )


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user information.

    This endpoint returns information about the currently authenticated user.
    Requires a valid JWT token in the Authorization header.

    Args:
        current_user: Current authenticated user (injected by dependency)

    Returns:
        Current user information

    Example:
        GET /api/v1/auth/me
        Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...

        Response:
        {
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "disabled": false
        }
    """
    if not config.api.enable_auth:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication is disabled in configuration"
        )

    return current_user


@router.post("/validate")
async def validate_token(current_user: User = Depends(get_current_active_user)):
    """
    Validate a JWT token.

    This endpoint validates the provided JWT token and returns success if valid.

    Args:
        current_user: Current authenticated user (injected by dependency)

    Returns:
        Validation status

    Example:
        POST /api/v1/auth/validate
        Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...

        Response:
        {
            "valid": true,
            "username": "admin"
        }
    """
    if not config.api.enable_auth:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication is disabled in configuration"
        )

    return {
        "valid": True,
        "username": current_user.username
    }


@router.get("/status")
async def auth_status():
    """
    Get authentication system status.

    This endpoint returns whether authentication is enabled or disabled.

    Returns:
        Authentication status

    Example:
        GET /api/v1/auth/status

        Response:
        {
            "enabled": false,
            "message": "Authentication is disabled"
        }
    """
    return {
        "enabled": config.api.enable_auth,
        "message": "Authentication is enabled" if config.api.enable_auth else "Authentication is disabled"
    }
