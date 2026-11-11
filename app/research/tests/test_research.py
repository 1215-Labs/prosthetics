"""Unit tests for research feature schemas, service, and routes.

These tests use mocks and an in-memory SQLite database so that no real
PostgreSQL connection is required.  Integration tests that require a live
database are decorated with ``@pytest.mark.integration``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.research.schemas import (
    LLMAnalysisRequest,
    LLMAnalysisResponse,
    ResearchPaperCreate,
    ResearchPaperUpdate,
)

# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestResearchPaperCreate:
    def test_title_required(self):
        with pytest.raises(ValueError):
            ResearchPaperCreate(title="")  # empty string fails min_length=1

    def test_tags_normalised(self):
        paper = ResearchPaperCreate(title="T", tags="  EMG , Hand , emg ")
        assert paper.tags == "emg,hand"

    def test_tags_empty_ok(self):
        paper = ResearchPaperCreate(title="T", tags="")
        assert paper.tags == ""

    def test_doi_optional(self):
        paper = ResearchPaperCreate(title="T")
        assert paper.doi is None

    def test_all_fields(self):
        paper = ResearchPaperCreate(
            title="Myoelectric hand control",
            abstract="We study...",
            authors="Smith J, Jones A",
            doi="10.1109/TNSRE.2021.12345",
            publication_year=2021,
            journal="IEEE TNSRE",
            tags="myoelectric,hand",
            source_url="https://example.com/paper",
        )
        assert paper.title == "Myoelectric hand control"
        assert paper.doi == "10.1109/TNSRE.2021.12345"


class TestResearchPaperUpdate:
    def test_all_fields_optional(self):
        update = ResearchPaperUpdate()
        assert update.title is None
        assert update.tags is None

    def test_tags_normalised_on_update(self):
        update = ResearchPaperUpdate(tags="  Prosthetic , Limb , prosthetic ")
        assert update.tags == "prosthetic,limb"

    def test_tags_none_untouched(self):
        update = ResearchPaperUpdate(tags=None)
        assert update.tags is None


class TestLLMAnalysisRequest:
    def test_valid(self):
        req = LLMAnalysisRequest(
            paper_id=1, question="What are the key findings?"
        )
        assert req.paper_id == 1

    def test_empty_question_fails(self):
        with pytest.raises(ValueError):
            LLMAnalysisRequest(paper_id=1, question="")


# ---------------------------------------------------------------------------
# Route tests (using TestClient with mocked service layer)
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)


def _make_mock_paper(paper_id: int = 1) -> MagicMock:
    """Create a MagicMock that mimics a ResearchPaper ORM instance."""
    paper = MagicMock()
    paper.id = paper_id
    paper.title = "Test Paper"
    paper.abstract = "An abstract."
    paper.authors = "Author A"
    paper.doi = None
    paper.publication_year = 2023
    paper.journal = "Test Journal"
    paper.tags = "prosthetics"
    paper.source_url = None
    paper.created_at = "2023-01-01T00:00:00+00:00"
    paper.updated_at = "2023-01-01T00:00:00+00:00"
    # Ensure model_fields_set and model_validate work via dict conversion
    paper.__dict__.update(
        {
            "id": paper_id,
            "title": "Test Paper",
            "abstract": "An abstract.",
            "authors": "Author A",
            "doi": None,
            "publication_year": 2023,
            "journal": "Test Journal",
            "tags": "prosthetics",
            "source_url": None,
            "created_at": "2023-01-01T00:00:00+00:00",
            "updated_at": "2023-01-01T00:00:00+00:00",
        }
    )
    return paper


class TestCreatePaperRoute:
    def test_create_paper_returns_201(self, client: TestClient) -> None:
        mock_paper = _make_mock_paper()
        with patch(
            "app.research.routes.service.create_paper",
            new_callable=AsyncMock,
            return_value=mock_paper,
        ):
            response = client.post(
                "/research/papers",
                json={"title": "Test Paper", "abstract": "An abstract."},
            )
        assert response.status_code == 201
        assert response.json()["title"] == "Test Paper"

    def test_create_paper_missing_title_returns_422(self, client: TestClient) -> None:
        response = client.post("/research/papers", json={"abstract": "No title"})
        assert response.status_code == 422


class TestListPapersRoute:
    def test_list_papers_returns_200(self, client: TestClient) -> None:
        mock_paper = _make_mock_paper()
        with patch(
            "app.research.routes.service.list_papers",
            new_callable=AsyncMock,
            return_value=([mock_paper], 1),
        ):
            response = client.get("/research/papers")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1

    def test_list_papers_with_tag_filter(self, client: TestClient) -> None:
        mock_paper = _make_mock_paper()
        with patch(
            "app.research.routes.service.list_papers",
            new_callable=AsyncMock,
            return_value=([mock_paper], 1),
        ) as mock_list:
            response = client.get("/research/papers?tag=prosthetics")
        assert response.status_code == 200
        mock_list.assert_called_once()
        _, kwargs = mock_list.call_args
        assert kwargs.get("tag") == "prosthetics"


class TestGetPaperRoute:
    def test_get_existing_paper(self, client: TestClient) -> None:
        mock_paper = _make_mock_paper(paper_id=5)
        with patch(
            "app.research.routes.service.get_paper",
            new_callable=AsyncMock,
            return_value=mock_paper,
        ):
            response = client.get("/research/papers/5")
        assert response.status_code == 200
        assert response.json()["id"] == 5

    def test_get_nonexistent_paper_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.research.routes.service.get_paper",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get("/research/papers/999")
        assert response.status_code == 404


class TestUpdatePaperRoute:
    def test_patch_existing_paper(self, client: TestClient) -> None:
        mock_paper = _make_mock_paper()
        updated = _make_mock_paper()
        updated.title = "Updated Title"
        updated.__dict__["title"] = "Updated Title"
        with (
            patch(
                "app.research.routes.service.get_paper",
                new_callable=AsyncMock,
                return_value=mock_paper,
            ),
            patch(
                "app.research.routes.service.update_paper",
                new_callable=AsyncMock,
                return_value=updated,
            ),
        ):
            response = client.patch(
                "/research/papers/1", json={"title": "Updated Title"}
            )
        assert response.status_code == 200
        assert response.json()["title"] == "Updated Title"

    def test_patch_nonexistent_paper_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.research.routes.service.get_paper",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.patch(
                "/research/papers/999", json={"title": "X"}
            )
        assert response.status_code == 404


class TestDeletePaperRoute:
    def test_delete_existing_paper_returns_204(self, client: TestClient) -> None:
        mock_paper = _make_mock_paper()
        with (
            patch(
                "app.research.routes.service.get_paper",
                new_callable=AsyncMock,
                return_value=mock_paper,
            ),
            patch(
                "app.research.routes.service.delete_paper",
                new_callable=AsyncMock,
            ),
        ):
            response = client.delete("/research/papers/1")
        assert response.status_code == 204

    def test_delete_nonexistent_paper_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.research.routes.service.get_paper",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.delete("/research/papers/999")
        assert response.status_code == 404


class TestAnalysePaperRoute:
    def test_analyse_paper_returns_200(self, client: TestClient) -> None:
        mock_response = LLMAnalysisResponse(
            paper_id=1,
            question="What are key findings?",
            analysis="The study found ...",
            model="gpt-4o",
            provider="openai",
        )
        with patch(
            "app.research.routes.service.analyse_paper_with_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/research/papers/1/analyse",
                json={"paper_id": 1, "question": "What are key findings?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["analysis"] == "The study found ..."
        assert data["provider"] == "openai"

    def test_analyse_paper_id_mismatch_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/research/papers/1/analyse",
            json={"paper_id": 2, "question": "Question"},
        )
        assert response.status_code == 422

    def test_analyse_nonexistent_paper_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.research.routes.service.analyse_paper_with_llm",
            new_callable=AsyncMock,
            side_effect=ValueError("Research paper with id=99 not found."),
        ):
            response = client.post(
                "/research/papers/99/analyse",
                json={"paper_id": 99, "question": "Question"},
            )
        assert response.status_code == 404

    def test_analyse_llm_error_returns_503(self, client: TestClient) -> None:
        from app.core.llm import LLMClientError

        with patch(
            "app.research.routes.service.analyse_paper_with_llm",
            new_callable=AsyncMock,
            side_effect=LLMClientError("API error"),
        ):
            response = client.post(
                "/research/papers/1/analyse",
                json={"paper_id": 1, "question": "Question"},
            )
        assert response.status_code == 503
