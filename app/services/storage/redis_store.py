"""
Redis result store.

Persists job state (queued → processing → done/failed) so the
API can serve status polls and WebSocket push without blocking.

Key scheme:  job:{job_id}  →  JSON-serialized JobState
TTL:  configured via REDIS_TTL_SECONDS (default 1 hour)
"""
import json
import time
import redis.asyncio as aioredis
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import JobState, JobStatusEnum, GenerateResponse


def _client() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _key(job_id: str) -> str:
    return f"job:{job_id}"


async def create_job(job_id: str) -> JobState:
    """Insert a new job in QUEUED state."""
    settings = get_settings()
    now = time.time()
    state = JobState(
        job_id=job_id,
        status=JobStatusEnum.queued,
        created_at=now,
        updated_at=now,
    )
    async with _client() as r:
        await r.set(
            _key(job_id),
            state.model_dump_json(),
            ex=settings.redis_ttl_seconds,
        )
    logger.info(f"Job created: {job_id}")
    return state


async def set_processing(job_id: str) -> None:
    """Mark job as processing."""
    await _update(job_id, status=JobStatusEnum.processing)
    logger.info(f"Job processing: {job_id}")


async def set_done(job_id: str, result: GenerateResponse) -> None:
    """Store completed result."""
    await _update(job_id, status=JobStatusEnum.done, result=result)
    logger.info(f"Job done: {job_id}")


async def set_failed(job_id: str, error: str) -> None:
    """Store failure reason."""
    await _update(job_id, status=JobStatusEnum.failed, error=error)
    logger.error(f"Job failed: {job_id} — {error}")


async def get_job(job_id: str) -> JobState | None:
    """Fetch current job state. Returns None if not found / expired."""
    async with _client() as r:
        raw = await r.get(_key(job_id))
    if not raw:
        return None
    return JobState.model_validate_json(raw)


async def _update(
    job_id: str,
    status: JobStatusEnum,
    result: GenerateResponse | None = None,
    error: str | None = None,
) -> None:
    settings = get_settings()
    async with _client() as r:
        raw = await r.get(_key(job_id))
        if not raw:
            logger.warning(f"Job not found in Redis: {job_id}")
            return
        state = JobState.model_validate_json(raw)
        state.status = status
        state.updated_at = time.time()
        if result is not None:
            state.result = result
        if error is not None:
            state.error = error
        await r.set(
            _key(job_id),
            state.model_dump_json(),
            ex=settings.redis_ttl_seconds,
        )