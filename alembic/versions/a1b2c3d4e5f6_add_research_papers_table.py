"""add research_papers table

Revision ID: a1b2c3d4e5f6
Revises: e4a05b88d90b
Create Date: 2026-07-11 10:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "e4a05b88d90b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create research_papers table."""
    op.create_table(
        "research_papers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False, server_default=""),
        sa.Column("authors", sa.String(length=1000), nullable=False, server_default=""),
        sa.Column("doi", sa.String(length=200), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("journal", sa.String(length=300), nullable=False, server_default=""),
        sa.Column("tags", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doi"),
    )


def downgrade() -> None:
    """Drop research_papers table."""
    op.drop_table("research_papers")
