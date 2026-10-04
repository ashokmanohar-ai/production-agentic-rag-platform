import hashlib

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import app.security as security
from app.config import Settings
from app.persistence.database import Base
from app.persistence.entities import MembershipEntity


def test_api_key_role_header_cannot_elevate_without_membership(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    token = "service-secret"
    settings = Settings(
        auth_enabled=True,
        api_key_sha256=hashlib.sha256(token.encode()).hexdigest(),
        database_url="sqlite+pysqlite:///:memory:",
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    monkeypatch.setattr(security, "build_session_factory", lambda _: sessions)

    with pytest.raises(HTTPException) as exc:
        security.get_security_context(
            authorization=f"Bearer {token}",
            x_tenant_id="tenant-a",
            x_project_id="project-a",
            x_role="admin",
        )
    assert exc.value.status_code == 403


def test_api_key_role_comes_from_membership_not_header(monkeypatch) -> None:
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
    token = "service-secret"
    settings = Settings(
        auth_enabled=True,
        api_key_sha256=hashlib.sha256(token.encode()).hexdigest(),
        database_url="sqlite+pysqlite:///:memory:",
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    monkeypatch.setattr(security, "build_session_factory", lambda _: sessions)

    context = security.get_security_context(
        authorization=f"Bearer {token}",
        x_tenant_id="tenant-a",
        x_project_id="project-a",
        x_role="admin",
    )
    assert context.role == "reader"
