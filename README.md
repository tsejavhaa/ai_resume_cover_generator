# ResumeAI — Resume Analysis & Cover Letter Generator

AI-powered resume tailoring with sync (inline) and async (Kafka queue) pipelines, supporting both local (Ollama) and cloud (DeepSeek) LLM backends.

## Architecture

```
Client
  │
  ├─ POST /generate    ← Sync: waits for LLM response
  └─ POST /submit      ← Async: returns job_id immediately
          │
          ▼
    Kafka (resume-jobs topic)
          │
          ▼
    Worker Pool (3 consumers)
          │
    generator.generate()   ← same pipeline reused
          │
          ├─ Redis (job state + result)
          └─ WebSocket broadcast → client
```

## Project Structure

```
.
├── app/                          # FastAPI backend
│   ├── main.py                   # App entry — lifespan, CORS, static mount
│   ├── api/routes.py             # All HTTP & WebSocket endpoints
│   ├── core/config.py            # Pydantic-settings config from .env
│   ├── models/schemas.py         # All Pydantic request/response models
│   ├── kafka/
│   │   ├── producer.py           # Kafka publisher (fire-and-forget)
│   │   ├── consumer.py           # 3-worker consumer pool (background task)
│   │   └── websocket_manager.py  # Job→WebSocket connection registry
│   ├── prompts/templates.py      # LLM prompt templates (cover letter + tweaks)
│   └── services/
│       ├── parser.py             # Document parsing (PDF/DOCX/TXT → text)
│       ├── nlp_extractor.py      # NER extraction + keyword matching + role inference
│       ├── llm_backends.py       # Ollama & DeepSeek backend abstraction
│       ├── generator.py          # Main pipeline orchestrator
│       ├── exporter.py           # PDF/DOCX/Markdown export
│       ├── resume_generator.py   # Resume improvement pipeline
│       ├── history_service.py    # Redis-backed job history store
│       └── storage/redis_store.py# Redis job state persistence
├── frontend/index.html           # Single-page SPA (inline CSS/JS)
├── tests/
│   ├── test_phase1.py            # Parser, NLP, match score tests
│   └── test_phase2.py            # Kafka, Redis, WebSocket manager tests
├── docker-compose.yml            # 6 services: api, zk, kafka, redis, kafka-ui
├── Dockerfile                    # Python 3.11-slim container
├── requirements.txt              # Python dependencies
├── EXPLANATION.md                # Full architecture & design document
└── .env                          # Environment config (LLM backend, model, keys)
```

## Quick Start

```bash
cp .env.example .env
# Edit .env: set LLM_BACKEND, OLLAMA_MODEL, DEEPSEEK_API_KEY, etc.

docker compose up --build -d

# Ensure Ollama is running natively on the host with the model pulled:
ollama pull llama3.2:3b

# Verify all containers are healthy:
docker compose ps
```

**Note**: Ollama runs natively on the host (not in Docker) to leverage GPU acceleration on Apple Silicon. The API container connects via `host.docker.internal:11434`. Start `ollama serve` if not already running via launchd.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/generate` | POST | Sync: parse, analyze, generate cover letter + tweaks inline |
| `/api/v1/submit` | POST | Async: enqueue job via Kafka, return `job_id` |
| `/api/v1/preview` | POST | Parse uploaded resume, return preview + extracted profile |
| `/api/v1/status/{job_id}` | GET | Poll job state from Redis |
| `/api/v1/improve-resume` | POST | Sync: generate improved resume with new skills at beginner level |
| `/api/v1/history` | GET | List recent job history (cover letters + resume improvements) |
| `/api/v1/history/{entry_id}` | GET | Get full history entry detail |
| `/api/v1/ws/{job_id}` | WS | Real-time status push via WebSocket |
| `/api/v1/health` | GET | Check backend availability |
| `/api/v1/export/{job_id}/pdf` | GET | Download result as PDF |
| `/api/v1/export/{job_id}/docx` | GET | Download result as DOCX |
| `/api/v1/export/{job_id}/md` | GET | Download result as Markdown |
| `/api/v1/export/{job_id}/resume-pdf` | GET | Download improved resume as PDF (if resume type) |

### Sync (Cover Letter)
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -F "resume=@resume.pdf" \
  -F "job_description=Senior Python Engineer..." \
  -F "tone=professional"
```

### Resume Improvement
```bash
curl -X POST http://localhost:8000/api/v1/improve-resume \
  -F "resume=@resume.pdf" \
  -F "job_description=Senior Python Engineer..."
```

### Async (Kafka)
```bash
JOB=$(curl -s -X POST http://localhost:8000/api/v1/submit \
  -F "resume=@resume.pdf" \
  -F "job_description=..." | jq -r .job_id)

curl http://localhost:8000/api/v1/status/$JOB
```

## Data Flow

### Sync Path
```
Client → POST /api/v1/generate → generator.generate() → LLM → GenerateResponse → Client
```

### Async Path
```
Client → POST /api/v1/submit → Kafka (resume-jobs) → Consumer Pool (3 workers)
  → Redis (job state) → generator.generate() → Redis (result) → WebSocket → Client
Client polls GET /api/v1/status/{job_id} until "done"
```

### Resume Improvement Path
```
Client → POST /api/v1/improve-resume → parser → NLP → gap analysis
  → LLM (new resume with beginner-level skills) → section diff extraction
  → ImproveResumeResponse → history save → Client
```

## Pipeline Detail

The cover letter generator runs 6 steps:
1. **Parse** resume file → raw text via `parser.extract_text()`
2. **NLP extraction** → `ExtractedProfile` + `JobProfile` via BERT-NER + keyword matching
3. **Role skill inference** → detects expected role tools and finds gaps
4. **Match scoring** → weighted formula (70% required skills, 30% preferred)
5. **Cover letter** → builds prompt → calls LLM → returns tailored text
6. **Resume tweaks** → builds prompt with role-suggested skills → calls LLM → parses JSON

The resume improvement generator adds a prompt that instructs the LLM to insert missing skills at beginner/pre-intermediate level under existing experience sections, using realistic phrasing like "completed an online course", "gained foundational knowledge of", or "built a proof-of-concept using".

## Key Design Decisions

1. **Dual sync/async paths** — simple requests use sync (inline wait); heavy loads use async (Kafka queue with 3 workers + Redis state)
2. **Pluggable LLM backends** — abstract `LLMBackend` class supports both Ollama (local GPU) and DeepSeek (cloud). Backend-specific prompts: DeepSeek gets chain-of-thought reasoning, format examples, and stricter constraints; Ollama gets concise direct prompts suited to smaller models
3. **Role-based skill intelligence** — `ROLE_SKILL_MAP` knowledge base suggests modern tools even when not explicitly mentioned in the JD
4. **Lazy-loaded NER** — HuggingFace BERT-NER pipeline loads on first use and caches; falls back gracefully to keyword-only mode
5. **Redis-backed history** — all generation results (cover letters + resume improvements) stored with 30-day TTL, capped at 100 entries
6. **Single-page frontend** — vanilla HTML/CSS/JS, two-phase generation flow (action → settings → execute), history modal with full detail display

## Infrastructure

- **Docker Compose**: 5 containers — FastAPI app, Zookeeper, Kafka (7.6.0), Redis (7.2-alpine), Kafka UI (debug at :8080)
- **Ollama**: Runs natively on host (Apple Silicon GPU acceleration). Connects via `host.docker.internal:11434`
- **Kafka**: Topic `resume-jobs`, single partition, consumer group `resume-workers`, 3 concurrent workers
- **Redis**: Dual purpose — job state with 1-hour TTL + history store with 30-day TTL

## History Feature

All generation results are stored in Redis with:
- Key scheme: `history:<uuid>` (sorted set index for listing)
- TTL: 30 days, capped at 100 entries
- Types: `cover_letter` (cover letter + tweaks + scores) and `resume` (improved resume + section diffs + new skills)
- Access: `GET /api/v1/history?limit=N` for listing, `GET /api/v1/history/{id}` for full detail

## Running Tests

```bash
pytest tests/ -v
```

## LLM Configuration

| Variable | Description | Default |
|---|---|---|
| `LLM_BACKEND` | Backend selection (`ollama` or `deepseek`) | `ollama` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2:3b` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |
| `DEEPSEEK_BASE_URL` | DeepSeek API base URL | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | DeepSeek model name | `deepseek-chat` |
