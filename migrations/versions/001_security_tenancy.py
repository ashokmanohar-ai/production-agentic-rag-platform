"""security tenancy and audit schema"""
from alembic import op
import sqlalchemy as sa
revision = "001_security_tenancy"
down_revision = None
branch_labels = None
depends_on = None
def upgrade() -> None:
    op.add_column("knowledge_documents", sa.Column("tenant_id", sa.String(100), nullable=False, server_default="default"))
    op.add_column("knowledge_documents", sa.Column("project_id", sa.String(100), nullable=False, server_default="default"))
    op.create_index("ix_knowledge_documents_tenant_id", "knowledge_documents", ["tenant_id"])
    op.create_index("ix_knowledge_documents_project_id", "knowledge_documents", ["project_id"])
    op.create_table("audit_events", sa.Column("id", sa.String(36), primary_key=True), sa.Column("subject", sa.String(200), nullable=False), sa.Column("tenant_id", sa.String(100), nullable=False), sa.Column("project_id", sa.String(100), nullable=False), sa.Column("action", sa.String(100), nullable=False), sa.Column("resource_type", sa.String(100), nullable=False), sa.Column("resource_id", sa.String(200)), sa.Column("outcome", sa.String(30), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("security_memberships", sa.Column("id", sa.String(36), primary_key=True), sa.Column("subject", sa.String(200), nullable=False), sa.Column("tenant_id", sa.String(100), nullable=False), sa.Column("project_id", sa.String(100), nullable=False), sa.Column("role", sa.String(30), nullable=False), sa.Column("active", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("subject", "tenant_id", "project_id", name="uq_subject_tenant_project"))
def downgrade() -> None:
    op.drop_table("security_memberships")
    op.drop_table("audit_events")
    op.drop_index("ix_knowledge_documents_project_id", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_tenant_id", table_name="knowledge_documents")
    op.drop_column("knowledge_documents", "project_id")
    op.drop_column("knowledge_documents", "tenant_id")
