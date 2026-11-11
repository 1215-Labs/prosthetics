"""SQLAlchemy models for the research feature.

Each ``ResearchPaper`` record represents a single scientific publication or
study in the prosthetics domain.  The model stores enough bibliographic
metadata to support AI-powered analysis and search, and links papers to
user-defined tag sets stored as a plain comma-separated string.
"""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.shared.models import TimestampMixin


class ResearchPaper(Base, TimestampMixin):
    """A scientific publication in the prosthetics research domain.

    Attributes:
        id: Auto-incrementing primary key.
        title: Full title of the paper.
        abstract: Abstract / summary of the paper.
        authors: Comma-separated list of author names.
        doi: Digital Object Identifier (optional, should be unique when set).
        publication_year: Year the paper was published.
        journal: Journal or conference name.
        tags: Comma-separated research tags, e.g. ``"myoelectric,hand,EMG"``.
        source_url: URL to the full text or landing page (optional).
        created_at: Inherited from :class:`~app.shared.models.TimestampMixin`.
        updated_at: Inherited from :class:`~app.shared.models.TimestampMixin`.
    """

    __tablename__ = "research_papers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False, default="")
    authors: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    doi: Mapped[str | None] = mapped_column(String(200), nullable=True, unique=True)
    publication_year: Mapped[int | None] = mapped_column(nullable=True)
    journal: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    tags: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
