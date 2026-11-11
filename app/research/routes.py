"""API routes for the research feature.

Exposes CRUD endpoints for research papers and an LLM-powered analysis
endpoint.  All routes are mounted under the ``/research`` prefix.

Endpoints
---------
- ``POST /research/papers`` - create a new paper
- ``GET  /research/papers`` - paginated list with optional tag filter
- ``GET  /research/papers/{paper_id}`` - fetch a single paper
- ``PATCH /research/papers/{paper_id}`` - partial update
- ``DELETE /research/papers/{paper_id}`` - delete
- ``POST /research/papers/{paper_id}/analyse`` - LLM analysis
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.llm import LLMClientError
from app.core.logging import get_logger
from app.research import service
from app.research.schemas import (
    LLMAnalysisRequest,
    LLMAnalysisResponse,
    ResearchPaperCreate,
    ResearchPaperResponse,
    ResearchPaperUpdate,
)
from app.shared.schemas import PaginatedResponse, PaginationParams

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post(
    "/papers",
    response_model=ResearchPaperResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_paper(
    payload: ResearchPaperCreate,
    db: AsyncSession = Depends(get_db),
) -> ResearchPaperResponse:
    """Create a new prosthetics research paper record.

    Args:
        payload: Paper metadata to store.
        db: Database session dependency.

    Returns:
        The newly created paper.
    """
    paper = await service.create_paper(db, payload)
    return ResearchPaperResponse.model_validate(paper)


@router.get("/papers", response_model=PaginatedResponse[ResearchPaperResponse])
async def list_papers(
    pagination: PaginationParams = Depends(),
    tag: str | None = Query(default=None, description="Filter by tag (substring match)"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ResearchPaperResponse]:
    """Return a paginated list of research papers.

    Args:
        pagination: Page number and page size.
        tag: Optional tag filter.
        db: Database session dependency.

    Returns:
        Paginated response containing paper records.
    """
    papers, total = await service.list_papers(
        db,
        offset=pagination.offset,
        limit=pagination.page_size,
        tag=tag,
    )
    return PaginatedResponse[ResearchPaperResponse](
        items=[ResearchPaperResponse.model_validate(p) for p in papers],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/papers/{paper_id}", response_model=ResearchPaperResponse)
async def get_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
) -> ResearchPaperResponse:
    """Fetch a single research paper by ID.

    Args:
        paper_id: Primary key of the paper.
        db: Database session dependency.

    Returns:
        The research paper.

    Raises:
        HTTPException: 404 if the paper does not exist.
    """
    paper = await service.get_paper(db, paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper with id={paper_id} not found.",
        )
    return ResearchPaperResponse.model_validate(paper)


@router.patch("/papers/{paper_id}", response_model=ResearchPaperResponse)
async def update_paper(
    paper_id: int,
    payload: ResearchPaperUpdate,
    db: AsyncSession = Depends(get_db),
) -> ResearchPaperResponse:
    """Partially update a research paper.

    Only fields that are explicitly supplied in the request body are changed.

    Args:
        paper_id: Primary key of the paper to update.
        payload: Fields to update.
        db: Database session dependency.

    Returns:
        The updated research paper.

    Raises:
        HTTPException: 404 if the paper does not exist.
    """
    paper = await service.get_paper(db, paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper with id={paper_id} not found.",
        )
    updated = await service.update_paper(db, paper, payload)
    return ResearchPaperResponse.model_validate(updated)


@router.delete("/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a research paper.

    Args:
        paper_id: Primary key of the paper to delete.
        db: Database session dependency.

    Raises:
        HTTPException: 404 if the paper does not exist.
    """
    paper = await service.get_paper(db, paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper with id={paper_id} not found.",
        )
    await service.delete_paper(db, paper)


@router.post("/papers/{paper_id}/analyse", response_model=LLMAnalysisResponse)
async def analyse_paper(
    paper_id: int,
    payload: LLMAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> LLMAnalysisResponse:
    """Use an LLM to analyse a prosthetics research paper.

    Builds a context-rich prompt from the paper's metadata and abstract,
    then submits the user's question to the configured LLM provider.

    Args:
        paper_id: Primary key of the paper to analyse (must match body).
        payload: Analysis request containing the scientific question.
        db: Database session dependency.

    Returns:
        LLM-generated analysis with model and provider metadata.

    Raises:
        HTTPException: 404 if the paper does not exist.
        HTTPException: 422 if paper_id in path does not match body.
        HTTPException: 503 if the LLM provider is unavailable or not configured.
    """
    if payload.paper_id != paper_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="paper_id in path must match paper_id in request body.",
        )

    try:
        return await service.analyse_paper_with_llm(db, paper_id, payload.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except LLMClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {exc}",
        ) from exc
