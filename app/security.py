from dataclasses import dataclass
import hashlib
import hmac
from typing import Literal

from fastapi import Header, HTTPException

from app.config import get_settings

Role = Literal["reader", "contributor", "admin"]


@dataclass(frozen=True, slots=True)
class SecurityContext:
    subject: str
    tenant_id: str
    project_id: str
    role: Role


_ROLE_LEVEL = {"reader": 1, "contributor": 2, "admin": 3}


def require_role(context: SecurityContext, minimum: Role) -> None:
    if _ROLE_LEVEL[context.role] < _ROLE_LEVEL[minimum]:
        raise HTTPException(status_code=403, detail="Insufficient role")


def get_security_context(
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
    x_project_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
) -> SecurityContext:
    settings = get_settings()
    if not settings.auth_enabled:
        return SecurityContext("development", x_tenant_id or "default", x_project_id or "default", "admin")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer API key required")
    presented = authorization.removeprefix("Bearer ").strip()
    expected_hash = settings.api_key_sha256
    if not expected_hash:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    digest = hashlib.sha256(presented.encode()).hexdigest()
    if not hmac.compare_digest(digest, expected_hash):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not x_tenant_id or not x_project_id:
        raise HTTPException(status_code=400, detail="Tenant and project headers are required")
    if x_role not in _ROLE_LEVEL:
        raise HTTPException(status_code=403, detail="Valid role header required")
    return SecurityContext("api-key", x_tenant_id, x_project_id, x_role)  # type: ignore[arg-type]
