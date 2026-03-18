# CLAUDE.md - Redis Agent Memory Server Project Context

> **This is a fork.** See [FORK.md](FORK.md) for what was changed, why, and how to rebase on upstream.
> Fork branch: `fork/openclaw-attribution` based on `server/v0.13.2`.
> Upstream: `https://github.com/redis/agent-memory-server`

## Redis Version
This project uses Redis 8, which is the redis:8 docker image.
Do not use Redis Stack or other earlier versions of Redis.

## Frequently Used Commands

### Project Setup
Get started in a new environment by installing `uv`:
```bash
pip install uv               # Install uv (once)
uv venv                      # Create a virtualenv (once)
uv install --all-extras      # Install dependencies
uv sync --all-extras         # Sync latest dependencies
```

### Activate the virtual environment
You MUST always activate the virtualenv before running commands:

```bash
source .venv/bin/activate
```

### Running Tests
Always run tests before committing. You MUST have 100% of the tests in the
code base passing to commit.

Run all tests like this, including tests that require API keys in the
environment:
```bash
uv run pytest --run-api-tests
```

Run fork-specific tests only (fast, no Redis required):
```bash
uv run pytest tests/test_attribution.py tests/test_forgetting.py tests/test_memory_vector_db.py tests/test_review_fixes.py -v
```

### Linting

```bash
uv run ruff check            # Run linting
uv run ruff format           # Format code
```

### Managing Dependencies
```bash
uv add <dependency>          # Add a dependency to pyproject.toml and update lock file
uv remove <dependency>       # Remove a dependency from pyproject.toml and update lock file
```

### Running Servers
```bash
uv run agent-memory api      # Start REST API server (default port 8000)
uv run agent-memory mcp      # Start MCP server (stdio mode)
uv run agent-memory mcp --mode sse --port 9000  # Start MCP server (SSE mode)
```

### Database Operations
```bash
uv run agent-memory rebuild-index     # Rebuild Redis search index
uv run agent-memory migrate-memories  # Run memory migrations
```

### Background Tasks
```bash
uv run agent-memory task-worker       # Start background task worker
uv run agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories"
```

### Docker Development
```bash
docker-compose up            # Start full stack (API, MCP, Redis)
docker-compose up redis      # Start only Redis Stack
docker-compose down          # Stop all services
```

### Committing Changes
IMPORTANT: This project uses `pre-commit`. You should run `pre-commit`
before committing:
```bash
uv run pre-commit install  # Install the hooks first
uv run pre-commit run --all-files
```

## Important Architectural Patterns

### Dual Interface Design (REST + MCP)
- **REST API**: Traditional HTTP endpoints for web applications (`api.py`)
- **MCP Server**: Model Context Protocol for AI agent integration (`mcp.py`)
- Both interfaces share the same core memory management logic

### Memory Architecture
```python
# Two-tier memory system
Working Memory (Session-scoped)  ->  Long-term Memory (Persistent)
    |                                      |
- Messages                          - Semantic search (vector + BM25 hybrid)
- Context                          - Topic modeling (controlled vocabulary)
- Structured memories              - Entity recognition (quality-filtered)
- Metadata                         - Hash-based deduplication
                                   - Multi-user attribution
```

### RedisVL Integration
**CRITICAL**: Always use RedisVL query types instead of direct redis-py client access for searches:
```python
# Correct - Use RedisVL queries
from redisvl.query import VectorQuery, FilterQuery
query = VectorQuery(vector=embedding, vector_field_name="vector", return_fields=["text"])

# Avoid - Direct redis client searches
# redis.ft().search(...)  # Don't do this
```

### Async-First Design
- All core operations are async
- Background task processing with Docket (DISABLED in production — USE_DOCKET=false)
- Async Redis connections throughout

## Fork-Specific Architecture

### Attribution Fields
All memories carry `source_user`, `source_channel`, `visibility`, and `stale_after`. These are:
- Stored as Redis TAG/NUMERIC fields in the index schema
- Filterable via `SourceUser`, `SourceChannel`, `VisibilityFilter`, `StaleAfter` filter types
- Propagated through merge (`merge_memories_with_llm`) and extraction (`_resolve_parent_attribution`)
- Returned in all search/list results including server-side recency queries

### Visibility Ranking
`VISIBILITY_RANK` in `models.py` is the single source of truth for visibility ordering:
```python
VISIBILITY_RANK = {"everyone": 0, "family": 1, "restricted": 2, "private": 3, "parents": 4, "admin": 5}
```
Both `extraction.py` and `long_term_memory.py` import this — never define inline copies.

### Hybrid Search
`search_memories()` runs both vector KNN and text BM25 searches in parallel, merging via Reciprocal Rank Fusion. Controlled by `hybrid_search_enabled` in config. Falls back gracefully if text search fails.

### Safety Defaults (Changed from Upstream)
- **Semantic dedup**: DISABLED (`semantic_dedup_enabled=False`). Was destructive and non-deterministic.
- **Automatic compaction**: DISABLED (`compaction_every_minutes=0`). Can still be triggered manually.
- **Hash-based dedup**: ACTIVE. Safe and deterministic.
- **Size guards**: `MAX_MEMORY_INPUT_CHARS=500`, `MAX_MEMORY_OUTPUT_CHARS=1000`, `MAX_ENTITY_COUNT=30`.

### Telemetry
OTLP HTTP metrics to SigNoz via `telemetry.py`. Uses `httpx`. Instruments embeddings, search, and store operations. Prefix: `memory_server.*`. Flush interval: 30s. Non-blocking — buffer is drained under lock but the HTTP POST happens outside the lock so callers of `record_metric()` are never stalled by a slow OTLP endpoint.

### Entity/Topic Quality
- `clean_entities()`: Stop word removal, URL/path/hex filtering, variant dedup, capped at 30
- `enforce_topics()`: Controlled topic taxonomy with synonym mapping from `~/.openclaw/config/memory-vocabulary.json` (deterministic longest-first substring matching)

## Critical Rules

### Import Placement
Place all imports at the top of modules, not inside functions. Inline imports should only be used when strictly necessary (e.g., avoiding circular dependencies, optional dependencies, or significant startup performance concerns).

### Authentication
- **PRODUCTION**: Never set `DISABLE_AUTH=true` in production
- **DEVELOPMENT**: Use `DISABLE_AUTH=true` for local testing only
- JWT/OAuth2 authentication required for all endpoints except `/health`, `/docs`, `/openapi.json`

### Memory Management
- Working memory automatically promotes structured memories to long-term storage
- Conversations are summarized when exceeding window size
- Use model-aware token limits for context window management
- Never re-enable semantic dedup or automatic compaction without thorough testing

### RedisVL Usage (Required)
Always use RedisVL query types for any search operations. This is a project requirement.

## Testing Notes

The project uses `pytest` with `testcontainers` for Redis integration testing:

- `uv run pytest` - Run all tests
- `uv run pytest tests/unit/` - Unit tests only
- `uv run pytest tests/integration/` - Integration tests (require Redis)
- `uv run pytest -v` - Verbose output
- `uv run pytest --cov` - With coverage

Fork-specific test files:
- `tests/test_attribution.py` — 29 tests: attribution CRUD, merge, extraction, API, MCP
- `tests/test_forgetting.py` — 22 tests: stale_after, TTL, budget, pinned
- `tests/test_memory_vector_db.py` — 41 tests: hybrid search, RRF, factory, embeddings
- `tests/test_review_fixes.py` — Shared constants, mock propagation, telemetry

## Project Structure

```
agent_memory_server/
├── main.py              # FastAPI application entry point
├── api.py               # REST API endpoints (+ conflict detection)
├── mcp.py               # MCP server implementation
├── config.py            # Configuration management (+ safety defaults)
├── auth.py              # OAuth2/JWT authentication
├── models.py            # Pydantic data models (+ VISIBILITY_RANK, attribution fields)
├── working_memory.py    # Session-scoped memory management
├── long_term_memory.py  # Persistent memory with semantic search (+ merge attribution, size guards)
├── messages.py          # Message handling and formatting
├── summarization.py     # Conversation summarization
├── extraction.py        # Topic/entity extraction (+ vocabulary, quality filtering, attribution)
├── filters.py           # Search filtering logic (+ SourceUser, SourceChannel, VisibilityFilter, StaleAfter)
├── telemetry.py         # OTLP HTTP metrics to SigNoz
├── llm/                 # LLM client package (LiteLLM-based)
│   ├── __init__.py      # Re-exports for clean imports
│   ├── client.py        # LLMClient class with chat/embedding methods
│   ├── embeddings.py    # LiteLLMEmbeddings (+ nomic prefix, telemetry)
│   ├── types.py         # ChatCompletionResponse, EmbeddingResponse, LLMBackend
│   └── exceptions.py    # LLMClientError, ModelValidationError, APIKeyMissingError
├── memory_vector_db.py  # Vector DB abstraction (+ hybrid search, attribution persistence)
├── memory_vector_db_factory.py  # DB factory (+ attribution index schema)
├── migrations.py        # Database schema migrations
├── docket_tasks.py      # Background task definitions
├── cli.py               # Command-line interface
├── dependencies.py      # FastAPI dependency injection
├── healthcheck.py       # Health check endpoint
├── logging.py           # Structured logging setup
├── client/              # Client libraries
└── utils/               # Utility modules
    ├── redis.py         # Redis connection and setup
    ├── redis_query.py   # RecencyAggregationQuery (+ attribution return fields)
    ├── recency.py       # Recency scoring and hashing
    ├── keys.py          # Redis key management
    └── api_keys.py      # API key utilities
```

## Core Components

### 1. Memory Management
- **Working Memory**: Session-scoped storage with automatic summarization
- **Long-term Memory**: Persistent storage with semantic search capabilities
- **Memory Promotion**: Automatic migration from working to long-term memory
- **Deduplication**: Hash-based only (semantic dedup disabled). Content hashing via SHA-256.
- **Attribution**: Every memory carries source_user, source_channel, visibility, stale_after

### 2. Search and Retrieval
- **Hybrid Search**: Vector KNN + text BM25 merged via Reciprocal Rank Fusion (RRF)
- **Filtering System**: Advanced filtering by session, namespace, topics, entities, timestamps, attribution
- **Recency Scoring**: Configurable semantic/recency weight blend (default 0.9/0.1)
- **RedisVL Integration**: All search operations use RedisVL query builders

### 3. AI Integration
- **Topic Modeling**: LLM-based extraction enforced to controlled topic vocabulary (loaded from config)
- **Entity Recognition**: LLM-based with quality filtering (stop words, dedup, cap at 30)
- **Summarization**: Conversation summarization when context window exceeded
- **Multi-LLM Support**: OpenAI, Anthropic, Ollama, and other LiteLLM providers

### 4. Authentication & Security
- **OAuth2/JWT**: Industry-standard authentication with JWKS validation
- **Multi-Provider**: Auth0, AWS Cognito, Okta, Azure AD support
- **Role-Based Access**: Fine-grained permissions using JWT claims
- **Development Mode**: `DISABLE_AUTH` for local development

### 5. Background Processing
- **Docket Tasks**: Redis-based task queue (DISABLED in production via USE_DOCKET=false)
- **Memory Indexing**: Asynchronous embedding generation and indexing
- **Compaction**: DISABLED by default. Manual trigger via /compact endpoint.

## Environment Configuration

Key environment variables:
```bash
# Redis
REDIS_URL=redis://localhost:6379

# Authentication (Production)
OAUTH2_ISSUER_URL=https://your-auth-provider.com
OAUTH2_AUDIENCE=your-api-audience
DISABLE_AUTH=false  # Never true in production

# Development
DISABLE_AUTH=true   # Local development only
LOG_LEVEL=DEBUG

# AI Services
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GENERATION_MODEL=gpt-5.4
EMBEDDING_MODEL=ollama/nomic-embed-text
OLLAMA_API_BASE=http://localhost:11434

# Memory Configuration
LONG_TERM_MEMORY=true
ENABLE_TOPIC_EXTRACTION=true
ENABLE_NER=true
USE_DOCKET=false
REDISVL_VECTOR_DIMENSIONS=768

# Safety (fork defaults)
SEMANTIC_DEDUP_ENABLED=false
COMPACTION_EVERY_MINUTES=0
HYBRID_SEARCH_ENABLED=true

# Telemetry
OTLP_METRICS_ENDPOINT=http://localhost:4318/v1/metrics
MEMORY_SERVER_TELEMETRY=true
```

## API Reference

### REST API (Port 8000)
- Session management (`/v1/working-memory/`)
- Working memory operations (`/v1/working-memory/{id}`)
- Long-term memory search (`/v1/long-term-memory/search`) — supports attribution filters
- Long-term memory create (`/v1/long-term-memory/`) — supports `?detect_conflicts=true`
- Long-term memory edit (`/v1/long-term-memory/{id}`) — supports attribution field updates
- Memory hydration (`/v1/memory/prompt`)

### MCP Server (Port 9000)
- `create_long_term_memories` - Store persistent memories (with attribution)
- `search_long_term_memory` - Semantic search with attribution filtering
- `edit_long_term_memory` - Update memory fields including attribution
- `memory_prompt` - Hydrate queries with relevant context
- `set_working_memory` - Manage session memory

## Development Workflow

0. **Install uv**: `pip install uv` to get started with uv
1. **Setup**: `uv install` to install dependencies
2. **Redis**: Start Redis via `docker-compose up redis`
3. **Development**: Use `DISABLE_AUTH=true` for local testing
4. **Testing**: Run `uv run pytest` before committing
5. **Linting**: Pre-commit hooks handle code formatting
6. **Deploy**: `launchctl kickstart -k gui/$(id -u)/ai.openclaw.memory-server`

## Documentation
- API docs available at `/docs` when server is running
- OpenAPI spec at `/openapi.json`
- Fork documentation in `FORK.md`
- Authentication examples in README.md
- System architecture diagram in `diagram.png`

# currentDate
Today's date is 2026-03-10.

      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
