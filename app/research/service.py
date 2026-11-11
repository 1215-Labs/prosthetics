"""Business logic for the research feature.

Service functions are async and accept an :class:`AsyncSession` as their
first argument so that they can be easily unit-tested by passing a mock or
test session directly.
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import LLMClientError, get_llm_client
from app.core.logging import get_logger
from app.research.models import ResearchPaper
from app.research.schemas import (
    LLMAnalysisResponse,
    ResearchPaperCreate,
    ResearchPaperUpdate,
)

logger = get_logger(__name__)

_PROSTHETICS_SYSTEM_PROMPT = (
    "You are an expert research assistant specialising in prosthetics and "
    "prosthetic devices.  Your role is to help scientists and engineers "
    "understand, critique, and build upon published research in this domain.  "
    "Provide concise, evidence-based answers grounded in the provided paper "
    "context.  Highlight key findings, limitations, and open research questions "
    "where relevant."
)


async def create_paper(
    db: AsyncSession,
    payload: ResearchPaperCreate,
) -> ResearchPaper:
    """Persist a new research paper record.

    Args:
        db: Active database session.
        payload: Validated create schema.

    Returns:
        The newly created :class:`ResearchPaper` ORM instance.
    """
    paper = ResearchPaper(**payload.model_dump())
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    logger.info(
        "research.paper.create_completed",
        paper_id=paper.id,
        title=paper.title[:80],
    )
    return paper


async def get_paper(db: AsyncSession, paper_id: int) -> ResearchPaper | None:
    """Retrieve a single research paper by primary key.

    Args:
        db: Active database session.
        paper_id: Primary key of the paper.

    Returns:
        The :class:`ResearchPaper` instance, or ``None`` if not found.
    """
    result = await db.execute(select(ResearchPaper).where(ResearchPaper.id == paper_id))
    return result.scalar_one_or_none()


async def list_papers(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 20,
    tag: str | None = None,
) -> tuple[list[ResearchPaper], int]:
    """Return a paginated list of research papers.

    Args:
        db: Active database session.
        offset: Number of records to skip.
        limit: Maximum number of records to return.
        tag: Optional tag filter (case-insensitive, substring match on the
             comma-separated ``tags`` column).

    Returns:
        A 2-tuple of ``(items, total_count)``.
    """
    query = select(ResearchPaper)
    count_query = select(func.count()).select_from(ResearchPaper)

    if tag:
        tag_filter = ResearchPaper.tags.ilike(f"%{tag.lower()}%")
        query = query.where(tag_filter)
        count_query = count_query.where(tag_filter)

    total_result = await db.execute(count_query)
    total: int = total_result.scalar_one()

    query = query.order_by(ResearchPaper.id.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    items: list[ResearchPaper] = list(result.scalars().all())

    return items, total


async def update_paper(
    db: AsyncSession,
    paper: ResearchPaper,
    payload: ResearchPaperUpdate,
) -> ResearchPaper:
    """Apply a partial update to an existing research paper.

    Args:
        db: Active database session.
        paper: The ORM instance to update.
        payload: Validated partial-update schema.

    Returns:
        The updated :class:`ResearchPaper` instance.
    """
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(paper, field, value)

    await db.commit()
    await db.refresh(paper)
    logger.info(
        "research.paper.update_completed",
        paper_id=paper.id,
        fields_updated=list(update_data.keys()),
    )
    return paper


async def delete_paper(db: AsyncSession, paper: ResearchPaper) -> None:
    """Delete a research paper from the database.

    Args:
        db: Active database session.
        paper: The ORM instance to delete.
    """
    await db.delete(paper)
    await db.commit()
    logger.info("research.paper.delete_completed", paper_id=paper.id)


async def analyse_paper_with_llm(
    db: AsyncSession,
    paper_id: int,
    question: str,
) -> LLMAnalysisResponse:
    """Query an LLM about a specific research paper.

    Fetches the paper from the database, builds a context-rich prompt with
    the paper's title, abstract, and metadata, and submits a scientific
    question to the configured LLM provider.

    Args:
        db: Active database session.
        paper_id: Primary key of the paper to analyse.
        question: The scientific question or analysis instruction.

    Returns:
        :class:`LLMAnalysisResponse` containing the generated analysis.

    Raises:
        ValueError: If the paper does not exist.
        LLMClientError: If the LLM provider request fails.
    """
    paper = await get_paper(db, paper_id)
    if paper is None:
        raise ValueError(f"Research paper with id={paper_id} not found.")

    context_lines = [
        f"Title: {paper.title}",
        f"Authors: {paper.authors or 'Unknown'}",
        f"Journal: {paper.journal or 'Unknown'}",
        f"Year: {paper.publication_year or 'Unknown'}",
        f"DOI: {paper.doi or 'N/A'}",
        f"Tags: {paper.tags or 'N/A'}",
        "",
        "Abstract:",
        paper.abstract or "(no abstract provided)",
    ]
    context = "\n".join(context_lines)

    prompt = (
        f"Below is a research paper in the prosthetics domain.\n\n"
        f"{context}\n\n"
        f"Question / Task:\n{question}"
    )

    logger.info(
        "research.llm_analysis.started",
        paper_id=paper_id,
        question_length=len(question),
    )

    try:
        client = get_llm_client()
        response = await client.complete(
            prompt=prompt,
            system=_PROSTHETICS_SYSTEM_PROMPT,
        )
    except (LLMClientError, ValueError) as exc:
        logger.error(
            "research.llm_analysis.failed",
            paper_id=paper_id,
            exc_info=True,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise

    logger.info(
        "research.llm_analysis.completed",
        paper_id=paper_id,
        analysis_length=len(response.text),
    )

    return LLMAnalysisResponse(
        paper_id=paper_id,
        question=question,
        analysis=response.text,
        model=response.model,
        provider=response.provider,
    )
