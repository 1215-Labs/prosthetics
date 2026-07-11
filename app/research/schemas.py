"""Pydantic schemas for the research feature."""

from pydantic import BaseModel, Field, field_validator


class ResearchPaperBase(BaseModel):
    """Shared fields for research paper create and update schemas."""

    title: str = Field(..., min_length=1, max_length=500, description="Paper title")
    abstract: str = Field(default="", description="Abstract / summary")
    authors: str = Field(
        default="",
        max_length=1000,
        description="Comma-separated author names, e.g. 'Smith J, Jones A'",
    )
    doi: str | None = Field(
        default=None,
        max_length=200,
        description="Digital Object Identifier, e.g. '10.1109/TNSRE.2021.12345'",
    )
    publication_year: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year of publication",
    )
    journal: str = Field(
        default="",
        max_length=300,
        description="Journal or conference name",
    )
    tags: str = Field(
        default="",
        max_length=500,
        description="Comma-separated research tags, e.g. 'myoelectric,hand,EMG'",
    )
    source_url: str | None = Field(
        default=None,
        description="URL to full text or landing page",
    )

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, value: str) -> str:
        """Normalise tag string: strip whitespace, lowercase, de-duplicate."""
        if not value:
            return value
        parts = [t.strip().lower() for t in value.split(",") if t.strip()]
        # Preserve insertion order while de-duplicating
        seen: set[str] = set()
        unique: list[str] = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique.append(part)
        return ",".join(unique)


class ResearchPaperCreate(ResearchPaperBase):
    """Schema for creating a new research paper."""


class ResearchPaperUpdate(BaseModel):
    """Schema for partially updating an existing research paper.

    All fields are optional - only supplied fields are updated.
    """

    title: str | None = Field(default=None, min_length=1, max_length=500)
    abstract: str | None = None
    authors: str | None = Field(default=None, max_length=1000)
    doi: str | None = Field(default=None, max_length=200)
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    journal: str | None = Field(default=None, max_length=300)
    tags: str | None = Field(default=None, max_length=500)
    source_url: str | None = None

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, value: str | None) -> str | None:
        """Normalise tag string when provided."""
        if value is None or not value:
            return value
        parts = [t.strip().lower() for t in value.split(",") if t.strip()]
        seen: set[str] = set()
        unique: list[str] = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique.append(part)
        return ",".join(unique)


class ResearchPaperResponse(ResearchPaperBase):
    """Schema returned by the API for a research paper.

    Extends the base schema with server-managed fields.
    """

    model_config = {"from_attributes": True}

    id: int
    created_at: str
    updated_at: str

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def serialise_datetime(cls, value: object) -> str:
        """Convert datetime objects to ISO-8601 strings."""
        return str(value)


class LLMAnalysisRequest(BaseModel):
    """Request body for LLM-powered paper analysis."""

    paper_id: int = Field(..., description="ID of the research paper to analyse")
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Scientific question or analysis instruction",
    )


class LLMAnalysisResponse(BaseModel):
    """Response from LLM-powered paper analysis."""

    paper_id: int
    question: str
    analysis: str
    model: str
    provider: str
