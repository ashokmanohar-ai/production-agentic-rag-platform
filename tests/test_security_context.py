import hashlib

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
import app.security as security_module
from app.persistence.database import Base
from app.persistence.entities import MembershipEntity
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
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    with sessions() as session:
        session.add(
            MembershipEntity(
                subject="api-key",
                tenant_id="tenant-a",
                project_id="project-a",
                role="reader",
                active=True,
            )
        )
        session.commit()
    monkeypatch.setattr(security_module, "build_session_factory", lambda _: sessions)
    context = get_security_context(f"Bearer {secret}", "tenant-a", "project-a", "admin")
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
