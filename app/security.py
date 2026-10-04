import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

from app.config import Settings, get_settings
from app.persistence.database import build_session_factory

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


def _jwt_subject(token: str, settings: Settings) -> str:
    if not settings.oidc_issuer or not settings.oidc_audience or not settings.oidc_jwks_url:
        raise HTTPException(status_code=503, detail="OIDC is not fully configured")
    try:
        key = PyJWKClient(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
        )
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise HTTPException(status_code=401, detail="JWT subject is missing")
        return subject
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid JWT") from exc


def _api_key_subject(token: str, settings: Settings) -> str:
    if not settings.api_key_sha256:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    digest = hashlib.sha256(token.encode()).hexdigest()
    if not hmac.compare_digest(digest, settings.api_key_sha256):
        raise HTTPException(status_code=401, detail="Invalid bearer credential")
    return "api-key"


def authenticate_subject(token: str, settings: Settings) -> str:
    if settings.oidc_enabled and token.count(".") == 2:
        return _jwt_subject(token, settings)
    return _api_key_subject(token, settings)


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
        raise HTTPException(status_code=401, detail="Bearer credential required")
    if not x_tenant_id or not x_project_id:
        raise HTTPException(status_code=400, detail="Tenant and project headers are required")
    token = authorization.removeprefix("Bearer ").strip()
    subject = authenticate_subject(token, settings)
    from app.persistence.membership import MembershipRepository

    role = MembershipRepository(build_session_factory(settings.database_url)).role_for(
        subject, x_tenant_id, x_project_id
    )
    if role is None:
        raise HTTPException(status_code=403, detail="No active tenant/project membership")
    return SecurityContext(subject, x_tenant_id, x_project_id, role)
