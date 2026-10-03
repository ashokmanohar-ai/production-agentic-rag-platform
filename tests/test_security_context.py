import hashlib

import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.security import SecurityContext, get_security_context, require_role


def test_development_context_defaults_to_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    context = get_security_context(None, "tenant-a", "project-a", None)
    assert context.tenant_id == "tenant-a"
    assert context.project_id == "project-a"
    assert context.role == "admin"
    get_settings.cache_clear()


def test_api_key_context_and_role(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret"
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY_SHA256", hashlib.sha256(secret.encode()).hexdigest())
    get_settings.cache_clear()
    context = get_security_context(f"Bearer {secret}", "tenant-a", "project-a", "reader")
    assert context == SecurityContext("api-key", "tenant-a", "project-a", "reader")
    require_role(context, "reader")
    with pytest.raises(HTTPException) as exc:
        require_role(context, "contributor")
    assert exc.value.status_code == 403
    get_settings.cache_clear()


def test_invalid_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY_SHA256", hashlib.sha256(b"right").hexdigest())
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as exc:
        get_security_context("Bearer wrong", "tenant-a", "project-a", "admin")
    assert exc.value.status_code == 401
    get_settings.cache_clear()
