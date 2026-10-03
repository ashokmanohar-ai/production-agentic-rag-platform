from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.entities import MembershipEntity
from app.security import Role


class MembershipRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def role_for(self, subject: str, tenant_id: str, project_id: str) -> Role | None:
        with self.sessions() as session:
            item = session.scalar(
                select(MembershipEntity).where(
                    MembershipEntity.subject == subject,
                    MembershipEntity.tenant_id == tenant_id,
                    MembershipEntity.project_id == project_id,
                    MembershipEntity.active.is_(True),
                )
            )
            if not item or item.role not in {"reader", "contributor", "admin"}:
                return None
            return item.role  # type: ignore[return-value]
