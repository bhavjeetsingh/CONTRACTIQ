"""
Rate Limiting for ContractIQ
=====================================
Prevents API abuse and controls LLM costs.

Limits:
- /analyze: 10 requests per minute per user
- /compare: 10 requests per minute per user  
- /chat/index: 5 requests per minute per user
- /chat/query: 20 requests per minute per user
- Global: 100 requests per minute per IP

Why this matters:
- Each LLM call costs money — unlimited requests = unlimited cost
- Prevents a single user from hammering the API
- Standard production requirement every interviewer expects
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Use IP address as the rate limit key for unauthenticated routes
# Use user email for authenticated routes
def get_user_identifier(request: Request) -> str:
    """
    Rate limit key function.
    Uses JWT user email if available, falls back to IP address.
    """
    # Try to get user from request state (set by JWT middleware)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "email"):
        return user.email
    # Fall back to IP address
    return get_remote_address(request)


# Global limiter instance
limiter = Limiter(key_func=get_user_identifier)
