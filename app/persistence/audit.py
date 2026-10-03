from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.entities import AuditEventEntity
from app.security import SecurityContext


class AuditRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def record(
        self,
        context: SecurityContext,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        outcome: str = "success",
    ) -> str:
        event = AuditEventEntity(
            subject=context.subject,
            tenant_id=context.tenant_id,
            project_id=context.project_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
        )
        with self.sessions() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event.id

    def list_for_scope(
        self, tenant_id: str, project_id: str, limit: int = 100
    ) -> list[AuditEventEntity]:
        with self.sessions() as session:
            items = list(session.scalars(
                select(AuditEventEntity)
                .where(
                    AuditEventEntity.tenant_id == tenant_id,
                    AuditEventEntity.project_id == project_id,
                )
                .order_by(AuditEventEntity.created_at.desc())
                .limit(limit)
            ))
            for item in items:
                session.expunge(item)
            return items
