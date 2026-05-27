"""
Job history service — persists all generation results in Redis.
Key scheme:  history:{entry_id}  →  JSON-serialized JobHistoryEntry
Sorted set:  history:index      →  scores = created_at, members = entry_id
TTL:  30 days (configurable via REDIS_HISTORY_TTL_DAYS)
"""
import json
import time
import uuid
import redis.asyncio as aioredis
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import (
    JobHistoryEntry, JobHistoryListItem, JobHistoryListResponse,
    GenerateResponse, ImproveResumeResponse,
)


def _client() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _key(entry_id: str) -> str:
    return f"history:{entry_id}"


INDEX_KEY = "history:index"
MAX_ENTRIES = 100


async def save_history(
    entry_type: str,
    filename: str,
    resume_text_preview: str,
    job_description: str,
    backend_used: str,
    model_used: str,
    computation_time_ms: int = 0,
    generate_response: GenerateResponse | None = None,
    improve_response: ImproveResumeResponse | None = None,
) -> JobHistoryEntry:
    settings = get_settings()
    now = time.time()
    entry_id = str(uuid.uuid4())

    entry = JobHistoryEntry(
        id=entry_id,
        type=entry_type,
        created_at=now,
        filename=filename,
        resume_text_preview=resume_text_preview,
        job_description=job_description,
        backend_used=backend_used,
        model_used=model_used,
        computation_time_ms=computation_time_ms,
    )

    if generate_response:
        entry.cover_letter = generate_response.cover_letter
        entry.resume_tweaks = generate_response.resume_tweaks
        entry.match_score = generate_response.match_score
        entry.matched_skills = generate_response.matched_skills
        entry.missing_skills = generate_response.missing_skills
        entry.role_suggested_skills = generate_response.role_suggested_skills

    if improve_response:
        entry.improved_resume_text = improve_response.improved_resume_text
        entry.improved_sections = improve_response.improved_sections
        entry.new_skills_added = improve_response.new_skills_added
        entry.match_score = improve_response.match_score
        entry.matched_skills = improve_response.matched_skills
        entry.missing_skills = improve_response.missing_skills
        entry.role_suggested_skills = improve_response.role_suggested_skills

    ttl_days = getattr(settings, "redis_history_ttl_days", 30)
    ttl_seconds = ttl_days * 86400

    async with _client() as r:
        await r.set(_key(entry_id), entry.model_dump_json(), ex=ttl_seconds)
        await r.zadd(INDEX_KEY, {entry_id: now})
        # Trim to last MAX_ENTRIES
        count = await r.zcard(INDEX_KEY)
        if count > MAX_ENTRIES:
            to_remove = count - MAX_ENTRIES
            old = await r.zrange(INDEX_KEY, 0, to_remove - 1)
            if old:
                await r.zrem(INDEX_KEY, *old)
                await r.delete(*[_key(e) for e in old])

    logger.info(f"History saved: {entry_id} type={entry_type}")
    return entry


async def get_history_entry(entry_id: str) -> JobHistoryEntry | None:
    async with _client() as r:
        raw = await r.get(_key(entry_id))
    if not raw:
        return None
    return JobHistoryEntry.model_validate_json(raw)


async def list_history(limit: int = 20) -> JobHistoryListResponse:
    async with _client() as r:
        ids = await r.zrevrange(INDEX_KEY, 0, limit - 1)
        entries: list[JobHistoryEntry] = []
        for eid in ids:
            raw = await r.get(_key(eid))
            if raw:
                try:
                    entries.append(JobHistoryEntry.model_validate_json(raw))
                except Exception:
                    continue
    items = [
        JobHistoryListItem(
            id=e.id,
            type=e.type,
            created_at=e.created_at,
            filename=e.filename,
            match_score=e.match_score,
            backend_used=e.backend_used,
            model_used=e.model_used,
        )
        for e in entries
    ]
    return JobHistoryListResponse(entries=items)
