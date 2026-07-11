# Prosthetics Research API

Public research platform for prosthetics and prosthetic devices, powered by FastAPI + PostgreSQL and AI/LLM-assisted scientific discovery.

## Purpose

This repository supports open scientific research in the domain of prosthetics and prosthetic devices.  Researchers can:

- **Manage research papers** – store and query bibliographic records with abstracts, authors, DOIs, and tags
- **AI-powered analysis** – submit scientific questions about any paper to an LLM (OpenAI or Anthropic) for expert-level analysis, critique, and discovery
- **Tag-based discovery** – filter papers by research tags (e.g. `myoelectric`, `hand`, `osseointegration`, `EMG`)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd prosthetics

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env – set DATABASE_URL and at least one LLM API key

# 4. Start services
docker-compose up -d

# 5. Run migrations
uv run alembic upgrade head

# 6. Start development server
uv run uvicorn app.main:app --reload --port 8123
```

Visit `http://localhost:8123/docs` for the interactive Swagger UI.

## API Endpoints

### Research Papers

| Method   | Path                                    | Description                         |
|----------|-----------------------------------------|-------------------------------------|
| `POST`   | `/research/papers`                      | Add a new research paper            |
| `GET`    | `/research/papers`                      | List papers (paginated, tag filter) |
| `GET`    | `/research/papers/{id}`                 | Fetch a single paper                |
| `PATCH`  | `/research/papers/{id}`                 | Partially update a paper            |
| `DELETE` | `/research/papers/{id}`                 | Delete a paper                      |
| `POST`   | `/research/papers/{id}/analyse`         | LLM-powered scientific analysis     |

### Health Checks

| Method | Path            | Description                    |
|--------|-----------------|--------------------------------|
| `GET`  | `/health`       | API health status              |
| `GET`  | `/health/db`    | Database connectivity check    |
| `GET`  | `/health/ready` | Full readiness check           |

## LLM Integration

The `/research/papers/{id}/analyse` endpoint sends a paper's title, abstract, and metadata to a configured LLM alongside a scientific question.  The LLM is prompted as a prosthetics research expert.

**Supported providers:**

- **OpenAI** – set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default: `gpt-4o`)
- **Anthropic** – set `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL` (default: `claude-opus-4-5`)

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` to select the active provider.

**Example request:**

```json
POST /research/papers/1/analyse
{
  "paper_id": 1,
  "question": "What are the main limitations of this study and how could future work address them?"
}
```

## Project Structure

```
app/
├── core/           # Infrastructure (config, database, logging, middleware, LLM client)
├── shared/         # Cross-feature utilities (pagination, timestamps)
├── research/       # Research papers feature slice
│   ├── models.py   # ResearchPaper SQLAlchemy model
│   ├── schemas.py  # Pydantic schemas
│   ├── service.py  # Business logic (CRUD + LLM analysis)
│   ├── routes.py   # FastAPI routes
│   └── tests/      # Unit tests
└── main.py         # FastAPI application entry point
```

## Database Providers

Works with any PostgreSQL provider:

```bash
# Docker (default)
DATABASE_URL=******localhost:5433/prosthetics_db

# Supabase
DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres

# Neon
DATABASE_URL=postgresql+asyncpg://[USER]:[PASSWORD]@[HOST].neon.tech/[DB]?sslmode=require

# Railway
DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@[HOST].railway.app:[PORT]/railway
```

## Commands

```bash
# Development
uv run uvicorn app.main:app --reload --port 8123

# Testing
uv run pytest -v                    # All tests (no DB required for unit tests)
uv run pytest -v -m integration     # Integration tests (requires live DB)

# Type checking
uv run mypy app/
uv run pyright app/

# Linting
uv run ruff check .
uv run ruff format .

# Database
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
uv run alembic downgrade -1

# Docker
docker-compose up -d                # Start services
docker-compose logs -f app          # View logs
docker-compose down                 # Stop services
```

## Tech Stack

- **Python 3.12+** with strict type checking (MyPy + Pyright)
- **FastAPI** – async web framework
- **PostgreSQL** – relational database (any provider)
- **SQLAlchemy 2.0** – async ORM
- **Alembic** – schema migrations
- **structlog** – structured JSON logging with request correlation
- **httpx** – async HTTP client for LLM API calls
- **uv** – fast Python package manager
- **Ruff** – linting and formatting
- **pytest** – testing

## License

MIT
