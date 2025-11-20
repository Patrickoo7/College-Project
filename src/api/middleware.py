"""Middleware components for API request processing.

This module provides middleware for:
- Authentication enforcement
- Request logging
- CORS handling (configured via app.py)
"""

from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Load configuration
config = get_config()


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce authentication on protected routes.

    This middleware checks if authentication is enabled in config and validates
    JWT tokens for protected routes (excluding public routes like /health, /docs, etc.).

    Note: With FastAPI, it's often better to use Depends() on individual routes
    rather than middleware for authentication. This middleware is provided as an
    example for cases where global authentication enforcement is needed.
    """

    # Routes that don't require authentication
    PUBLIC_ROUTES = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/status",
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request and enforce authentication if enabled.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler in the chain

        Returns:
            HTTP response
        """
        # Skip authentication check if disabled in config
        if not config.api.enable_auth:
            return await call_next(request)

        # Check if route is public
        path = request.url.path
        if self._is_public_route(path):
            return await call_next(request)

        # Check for Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning(f"Unauthorized access attempt to: {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Note: Actual token validation is handled by the auth dependency
        # in individual routes. This middleware only checks for token presence.

        # Process request
        response = await call_next(request)
        return response

    def _is_public_route(self, path: str) -> bool:
        """
        Check if a route is public (doesn't require authentication).

        Args:
            path: Request path

        Returns:
            True if route is public, False otherwise
        """
        # Check exact matches
        if path in self.PUBLIC_ROUTES:
            return True

        # Check prefix matches (for paths like /docs/*)
        for public_route in self.PUBLIC_ROUTES:
            if path.startswith(public_route):
                return True

        return False


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests and responses.

    This middleware logs:
    - Request method and path
    - Response status code
    - Request processing time
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request and log details.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler in the chain

        Returns:
            HTTP response
        """
        import time

        # Log incoming request
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"

        logger.debug(f"Incoming request: {method} {path} from {client_host}")

        # Process request and measure time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        status_code = response.status_code
        logger.info(
            f"Request completed: {method} {path} - "
            f"Status: {status_code} - "
            f"Time: {process_time:.3f}s"
        )

        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware (example implementation).

    This is a basic example of rate limiting. For production use,
    consider using a more robust solution like slowapi or Redis-based rate limiting.

    Note: This implementation uses in-memory storage and won't work
    across multiple worker processes.
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {ip: [(timestamp, count), ...]}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request and enforce rate limits.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler in the chain

        Returns:
            HTTP response or rate limit error
        """
        import time

        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Get client IP
        client_host = request.client.host if request.client else "unknown"

        # Check rate limit
        current_time = time.time()

        if client_host not in self.requests:
            self.requests[client_host] = []

        # Remove old requests outside the window
        self.requests[client_host] = [
            (ts, count) for ts, count in self.requests[client_host]
            if current_time - ts < self.window_seconds
        ]

        # Count requests in current window
        request_count = sum(count for _, count in self.requests[client_host])

        if request_count >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_host}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds} seconds."
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Add current request
        self.requests[client_host].append((current_time, 1))

        # Process request
        response = await call_next(request)
        return response
