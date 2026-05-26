"""
FastAPI application entry point — Phase 1 + Phase 2.
"""
import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

# ── Logging ──────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
)


# ── Lifespan: start Kafka consumer pool in background ─────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting Resume Generator | env={settings.app_env} "
        f"backend={settings.llm_backend}"
    )
    # Start Kafka consumer pool as a background task (non-blocking)
    consumer_task = asyncio.create_task(_start_kafka())
    yield
    # Shutdown — cancel consumer
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete")


async def _start_kafka():
    """Start consumer pool, swallowing errors if Kafka is unavailable."""
    try:
        from app.kafka.consumer import start_consumer_pool
        await start_consumer_pool()
    except Exception as e:
        logger.warning(
            f"Kafka consumer not started: {e}. "
            "Async /submit endpoint unavailable. Sync /generate still works."
        )


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Resume & Cover Letter Generator",
    description=(
        "AI-powered resume tailoring.\n\n"
        "**Sync:** `POST /api/v1/generate` — wait for result inline.\n\n"
        "**Async:** `POST /api/v1/submit` → poll `GET /api/v1/status/{job_id}` "
        "or connect `WS /api/v1/ws/{job_id}`."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Resume & Cover Letter Generator API v2",
        "docs": "/docs",
        "endpoints": {
            "sync": "POST /api/v1/generate",
            "async_submit": "POST /api/v1/submit",
            "async_status": "GET /api/v1/status/{job_id}",
            "async_ws": "WS /api/v1/ws/{job_id}",
            "health": "GET /api/v1/health",
        },
    }