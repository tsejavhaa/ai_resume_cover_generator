# ResumeAI — Project Architecture & Design

## Overview

ResumeAI is an intelligent resume analysis & cover letter generation system. A user uploads their resume and a job description; the app extracts structured data (skills, job titles, education), computes a match score, and uses an LLM (Ollama locally or DeepSeek via cloud) to generate a tailored cover letter and resume improvement suggestions.

The system supports both synchronous (wait inline) and asynchronous (Kafka queue + Redis state + WebSocket push) processing paths.

---

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
│       ├── generator.py          # Main pipeline orchestrator (6 steps)
│       ├── exporter.py           # PDF/DOCX/Markdown export
│       └── storage/redis_store.py# Redis job state persistence
├── frontend/index.html           # Single-page SPA (inline CSS/JS)
├── tests/
│   ├── test_phase1.py            # Parser, NLP, match score tests
│   └── test_phase2.py            # Kafka, Redis, WebSocket manager tests
├── docker-compose.yml            # 6 services: api, ollama, zk, kafka, redis, kafka-ui
├── Dockerfile                    # Python 3.11-slim container
├── requirements.txt              # Python dependencies
└── .env                          # Environment config (LLM backend, model, keys)
```

---

## File Purposes

### `app/main.py`
FastAPI application entry point. Creates the app, configures CORS (allow all origins for dev), registers a lifespan that starts the Kafka consumer pool as a background task, mounts the frontend static directory at `/ui`, and defines the root `/` endpoint listing all available routes.

### `app/api/routes.py`
All API endpoint definitions. 9 endpoints total:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/generate` | POST | Sync: parse, analyze, generate cover letter + tweaks inline |
| `/api/v1/submit` | POST | Async: enqueue job via Kafka, return `job_id` immediately |
| `/api/v1/preview` | POST | Parse uploaded resume and return preview text + extracted profile |
| `/api/v1/status/{job_id}` | GET | Poll job state from Redis |
| `/api/v1/ws/{job_id}` | WS | Real-time status push via WebSocket |
| `/api/v1/health` | GET | Check Ollama & DeepSeek backend availability |
| `/api/v1/export/{job_id}/pdf` | GET | Download result as PDF (via reportlab) |
| `/api/v1/export/{job_id}/docx` | GET | Download result as DOCX (via python-docx) |
| `/api/v1/export/{job_id}/md` | GET | Download result as Markdown |

### `app/core/config.py`
Singleton `Settings` class using `pydantic-settings` that reads from `.env`. Contains: LLM backend type, Ollama URL/model, DeepSeek API key/model, NER model name, Kafka bootstrap servers, Redis URL, number of Kafka workers, and Redis TTL.

### `app/models/schemas.py`
All Pydantic models used across the system:
- **Request**: `GenerateRequest` — form fields for generation
- **Kafka**: `JobMessage` — serialized payload published to Kafka topic `resume-jobs`
- **Redis**: `JobState` — persisted in Redis for status polling, TTL 1 hour
- **Internal**: `ExtractedProfile` (resume structured data), `JobProfile` (JD structured data)
- **Response**: `GenerateResponse` — cover letter + tweaks + scores + preview text, `PreviewResponse` — parsed resume preview before submission, `SubmitResponse`, `StatusResponse`, `ErrorResponse`
- **Enums**: `ToneEnum`, `BackendEnum`, `JobStatusEnum`

### `app/kafka/producer.py`
Async Kafka producer. The `publish_job()` function creates a one-shot `AIOKafkaProducer`, serializes a `JobMessage` to JSON, publishes to the `resume-jobs` topic with `job_id` as the partition key, then disconnects. Also provides `encode_file()`/`decode_file()` for base64 file encoding.

### `app/kafka/consumer.py`
Kafka consumer pool. `start_consumer_pool()` is called at app lifespan and spawns `KAFKA_NUM_WORKERS` (default 3) concurrent AIOKafkaConsumer tasks within a single consumer group (`resume-workers`). Each worker: deserializes the message → updates Redis to "processing" → calls `generator.generate()` → updates Redis to "done"/"failed" → broadcasts via WebSocket. Retries up to 10x for Kafka connection with exponential backoff.

### `app/kafka/websocket_manager.py`
WebSocket connection manager. Singleton class that maintains a dict of `job_id → set[WebSocket connections]`. Methods: `connect()`, `disconnect()`, `broadcast()` (sends JSON to all connections for a job). Used by both the WebSocket endpoint (on connect/disconnect) and the Kafka consumer (on job completion).

### `app/prompts/templates.py`
Versioned LLM prompt templates. Two template pairs:
1. **Cover Letter**: System prompt sets tone/length rules; user prompt provides resume + JD + skills context. `build_cover_letter_messages()` assembles the message array.
2. **Resume Tweaks**: System prompt specifies JSON output format (ATS optimization); user prompt includes resume, JD, candidate skills, missing skills, role-suggested skills, and detected role. `build_tweaks_messages()` assembles the message array.

### `app/services/parser.py`
Document parser. `extract_text()` dispatches by file extension:
- **PDF**: `pdfminer.high_level.extract_text_to_fp()` with LAParams
- **DOCX**: `python-docx` iterates paragraphs + table cells
- **TXT**: UTF-8 or Latin-1 decode
- `_clean()` normalizes whitespace (collapses 3+ blank lines to 2)

### `app/services/nlp_extractor.py`
Core NLP layer with 4 analysis functions:
1. **`extract_resume_profile(text)`** → `ExtractedProfile`: runs BERT-NER (`dslim/bert-base-NER`, lazy-loaded), extracts skills via keyword matching (30+ tech keywords), job titles via regex (5 patterns), organizations from NER, education from keyword lines
2. **`extract_job_profile(text)`** → `JobProfile`: same NER + skill extraction, splits skills into required/preferred based on signal regex matching, extracts company name and job title
3. **`compute_match_score(resume, job)`** → `(score, matched, missing)`: weighted formula — required skills contribute 70%, preferred 30%. Returns 0-100 score plus skill lists.
4. **`infer_role_skills(job_title, job_description)`** → `list[str]`: role-based skill intelligence. Matches the job title/JD against role detection patterns (ML engineer, data scientist, DevOps, etc.) and returns a curated set of modern skills expected for that role (e.g., "ML Engineer" → HuggingFace, PyTorch, scikit-learn, MLflow). This enables the system to suggest skills even when they aren't explicitly mentioned in the JD.

**Skill knowledge base**: A dictionary mapping 10 role categories to their expected modern tool/technology sets. The `ROLE_SKILL_MAP` includes roles like "machine learning", "data science", "devops", "backend", "ai engineer", etc.

### `app/services/llm_backends.py`
Abstraction layer over LLM providers:
- **`LLMBackend`**: Abstract base class with `chat(messages, temperature, max_tokens)` and `is_available()` methods
- **`OllamaBackend`**: Calls local Ollama API at `http://ollama:11434/api/chat` with 300s read timeout
- **`DeepSeekBackend`**: Calls DeepSeek's OpenAI-compatible `chat/completions` endpoint
- **`get_backend(override)`**: Factory that returns the configured backend (from `.env` or explicit override)

### `app/services/generator.py`
Main pipeline orchestrator. The `generate()` function runs 6 steps:
1. **Parse** resume file → raw text via `parser.extract_text()`
2. **NLP extraction** → `ExtractedProfile` + `JobProfile` via `nlp_extractor`
3. **Role skill inference** → `infer_role_skills()` detects expected role tools and finds gaps
4. **Match scoring** → `compute_match_score()` returns 0-100 score
5. **Cover letter** → builds prompt → calls LLM → returns text
6. **Resume tweaks** → builds prompt with role-suggested skills context → calls LLM → parses JSON → returns `ResumeTweak[]`

Returns `GenerateResponse` with cover letter, tweaks, scores, role-suggested skills, and a preview of the parsed resume text.

### `app/services/exporter.py`
Export to three formats:
- **PDF** via reportlab — styled document with score ring, color-coded sections, tables
- **DOCX** via python-docx — structured Word document
- **Markdown** — plain text with headings

### `app/services/storage/redis_store.py`
Async Redis operations using `redis.asyncio`. Key scheme: `job:{job_id}` → JSON-serialized `JobState`. Functions: `create_job()` (queued), `set_processing()`, `set_done()` (stores result), `set_failed()` (stores error), `get_job()`. All keys TTL 1 hour.

### `frontend/index.html`
Single-page monolithic frontend (632+ lines) with inline CSS and vanilla JS. No framework — pure HTML/CSS/JS. Dark theme with animated background grid. Logical sections:

1. **Upload zone** — drag-drop file selection with file badge
2. **Resume preview** (new) — auto-parses on file select, shows extracted text, skills, role, and expected skills
3. **Job description form** — textarea, tone/length/mode/backend selects
4. **Error banner** — validation and network errors
5. **Status bar** — spinner, progress bar, step indicators (6 animated steps)
6. **Results** — match score ring, skill pills (matched/missing/role-suggested), cover letter with copy + export, resume tweaks with before/after display, parsed resume preview

### `tests/test_phase1.py`
12 tests covering: parser text cleaning, PDF/DOCX/TXT parsing, skill extraction, job title extraction, education extraction, match score computation (perfect match, no overlap, boundary values), full profile extraction, and the complete pipeline.

### `tests/test_phase2.py`
7 tests covering: Kafka message UUID generation, base64 encode/decode round-trip, JSON serialization, Redis store CRUD operations, job state transitions, state serialization with result, WebSocket manager connection tracking and safe broadcast.

---

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

### Preview Path
```
Client selects file → POST /api/v1/preview → parser.extract_text() + nlp_extractor
  → PreviewResponse (text, skills, role, expected skills) → Frontend preview card
```

---

## Key Design Decisions

1. **Dual sync/async paths**: Simple requests use sync (inline wait); heavy loads use async (Kafka queue with 3 workers + Redis for state).

2. **Pluggable LLM backends**: Abstract `LLMBackend` class supports both local (Ollama) and cloud (DeepSeek) providers. New backends can be added by subclassing.

3. **Role-based skill intelligence**: The `ROLE_SKILL_MAP` knowledge base enables the system to suggest modern tools even when not explicitly mentioned in the JD — e.g., suggesting HuggingFace for an ML Engineer role.

4. **Lazy-loaded NER**: HuggingFace BERT-NER pipeline loads on first use and caches. Falls back gracefully to keyword-only mode if model unavailable.

5. **Prompt versioning**: All LLM prompts are in one file (`templates.py`) for easy iteration without touching business logic.

6. **Single-page frontend**: Vanilla HTML/CSS/JS avoids framework overhead while maintaining a polished dark UI with animated elements.

---

## Infrastructure

- **Docker Compose**: 6 containers — FastAPI app, Ollama, Zookeeper, Kafka (7.6.0), Redis (7.2-alpine), Kafka UI (debug at :8080)
- **Kafka**: Topic `resume-jobs`, single partition, consumer group `resume-workers`, 3 concurrent workers
- **Redis**: Key-value store for job state with 1-hour TTL, used by both status polling and WebSocket push
- **Ollama**: Local LLM serving, health-checked via `ollama list`, default model `llama3.2:1b`
