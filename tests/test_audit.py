from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.audit import AuditRepository
from app.persistence.database import Base
from app.security import SecurityContext


def test_audit_events_are_scoped() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    repo = AuditRepository(sessionmaker(bind=engine, expire_on_commit=False, class_=Session))
    context = SecurityContext("user-1", "tenant-a", "project-a", "admin")
    event_id = repo.record(context, "document.upload", "document", "doc-1")
    events = repo.list_for_scope("tenant-a", "project-a")
    assert len(events) == 1
    assert events[0].id == event_id
    assert repo.list_for_scope("tenant-b", "project-b") == []
