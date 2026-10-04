"""phase 16 evaluation tenancy and trace ownership"""

from alembic import op
import sqlalchemy as sa

revision = "002_security_closure"
down_revision = "001_security_tenancy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "evaluation_runs",
        sa.Column("tenant_id", sa.String(100), nullable=False, server_default="default"),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column("project_id", sa.String(100), nullable=False, server_default="default"),
    )
    op.create_index("ix_evaluation_runs_tenant_id", "evaluation_runs", ["tenant_id"])
    op.create_index("ix_evaluation_runs_project_id", "evaluation_runs", ["project_id"])
    op.create_table(
        "trace_ownership",
        sa.Column("trace_id", sa.String(64), primary_key=True),
        sa.Column("subject", sa.String(200), nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("project_id", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_trace_ownership_subject", "trace_ownership", ["subject"])
    op.create_index("ix_trace_ownership_tenant_id", "trace_ownership", ["tenant_id"])
    op.create_index("ix_trace_ownership_project_id", "trace_ownership", ["project_id"])
    op.create_index("ix_trace_ownership_created_at", "trace_ownership", ["created_at"])


def downgrade() -> None:
    op.drop_table("trace_ownership")
    op.drop_index("ix_evaluation_runs_project_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_tenant_id", table_name="evaluation_runs")
    op.drop_column("evaluation_runs", "project_id")
    op.drop_column("evaluation_runs", "tenant_id")
