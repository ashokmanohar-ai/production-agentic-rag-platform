from sqlalchemy.orm import Session, sessionmaker

from app.persistence.entities import TraceOwnershipEntity
from app.security import SecurityContext


class TraceOwnershipRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def record(self, trace_id: str, context: SecurityContext) -> None:
        with self.sessions() as session:
            session.merge(
                TraceOwnershipEntity(
                    trace_id=trace_id,
                    subject=context.subject,
                    tenant_id=context.tenant_id,
                    project_id=context.project_id,
                )
            )
            session.commit()

    def belongs_to_scope(self, trace_id: str, context: SecurityContext) -> bool:
        with self.sessions() as session:
            item = session.get(TraceOwnershipEntity, trace_id)
            return bool(
                item
                and item.tenant_id == context.tenant_id
                and item.project_id == context.project_id
            )
