# Resume & Cover Letter Generator — Phase 1 + 2

AI-powered resume tailoring with sync and async (Kafka) pipelines.

## Architecture

```
Client
  │
  ├─ POST /generate    ← Phase 1: sync, waits for LLM response
  └─ POST /submit      ← Phase 2: async, returns job_id immediately
          │
          ▼
    Kafka (resume-jobs topic)
          │
          ▼
    Worker Pool (3 consumers)
          │
    generator.generate()   ← same Phase 1 pipeline reused
          │
          ├─ Redis (job state + result)
          └─ WebSocket broadcast → client
```

## Project Structure

```
resume_generator/
├── app/
│   ├── api/
│   │   └── routes.py           # /generate /submit /status /ws /health
│   ├── core/
│   │   └── config.py           # all settings via pydantic-settings
│   ├── kafka/
│   │   ├── producer.py         # publish_job() → Kafka
│   │   ├── consumer.py         # worker pool, calls generator.generate()
│   │   └── websocket_manager.py# real-time push to connected clients
│   ├── models/
│   │   └── schemas.py          # all Pydantic models
│   ├── prompts/
│   │   └── templates.py        # versioned LLM prompt templates
│   ├── services/
│   │   ├── parser.py           # PDF/DOCX/TXT extraction
│   │   ├── nlp_extractor.py    # HuggingFace NER + skill matching
│   │   ├── llm_backends.py     # Ollama + DeepSeek (pluggable)
│   │   ├── generator.py        # main pipeline orchestrator
│   │   └── storage/
│   │       └── redis_store.py  # job state persistence
│   └── main.py                 # FastAPI app + Kafka consumer startup
├── tests/
│   ├── test_phase1.py          # NLP + parser unit tests
│   └── test_phase2.py          # Kafka schema + Redis state tests
├── docker-compose.yml          # API + Ollama + Kafka + Redis + Kafka UI
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
cp .env.example .env
# Edit .env: set LLM_BACKEND, OLLAMA_MODEL, etc.

docker compose up --build -d

# Pull model into Ollama container (once)
docker compose exec ollama ollama pull llama3.2:3b

# Warm up the model
docker compose exec ollama ollama run llama3.2:3b "say hi"
```

## API Usage

### Sync (Phase 1)
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -F "resume=@resume.pdf" \
  -F "job_description=Senior Python Engineer..." \
  -F "tone=professional"
```

### Async (Phase 2)
```bash
# 1. Submit — returns immediately
JOB=$(curl -s -X POST http://localhost:8000/api/v1/submit \
  -F "resume=@resume.pdf" \
  -F "job_description=Senior Python Engineer..." | jq -r .job_id)

# 2. Poll status
curl http://localhost:8000/api/v1/status/$JOB

# 3. Or connect WebSocket for real-time push
wscat -c "ws://localhost:8000/api/v1/ws/$JOB"
```

### Kafka UI (debug)
Open http://localhost:8080 — browse topics, messages, consumer groups.

## Running Tests
```bash
pytest tests/ -v
```

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | FastAPI sync pipeline — parse → NLP → LLM → response |
| 2 | ✅ Done | Kafka async queue + Redis state + WebSocket push |
| 3 | ⬜ Next | PDF/DOCX export, frontend UI |