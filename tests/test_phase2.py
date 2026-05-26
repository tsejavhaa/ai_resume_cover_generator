"""
Phase 2 tests — Kafka message schema and Redis store logic.
Run with: pytest tests/test_phase2.py -v

Note: Redis/Kafka integration tests require running services.
Unit tests here mock those dependencies.
"""
import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.models.schemas import (
    JobMessage, JobState, JobStatusEnum,
    GenerateResponse, ResumeTweak,
)
from app.kafka.producer import encode_file, decode_file


# ── Kafka message tests ───────────────────────────────────────

def test_job_message_generates_uuid():
    msg1 = JobMessage(
        resume_bytes_b64="abc",
        filename="resume.pdf",
        job_description="x" * 60,
        tone="professional",
        backend=None,
        cover_letter_length="medium",
    )
    msg2 = JobMessage(
        resume_bytes_b64="abc",
        filename="resume.pdf",
        job_description="x" * 60,
        tone="professional",
        backend=None,
        cover_letter_length="medium",
    )
    assert msg1.job_id != msg2.job_id


def test_encode_decode_file_roundtrip():
    original = b"PDF binary content \x00\x01\x02"
    encoded = encode_file(original)
    assert isinstance(encoded, str)
    decoded = decode_file(encoded)
    assert decoded == original


def test_job_message_serializes_to_json():
    msg = JobMessage(
        resume_bytes_b64=encode_file(b"test"),
        filename="cv.pdf",
        job_description="y" * 60,
        tone="confident",
        backend="ollama",
        cover_letter_length="short",
    )
    payload = json.dumps(msg.model_dump())
    restored = JobMessage(**json.loads(payload))
    assert restored.job_id == msg.job_id
    assert restored.tone == "confident"


# ── Redis store unit tests (mocked) ──────────────────────────

@pytest.fixture
def mock_redis():
    store = {}
    mock = AsyncMock()

    async def fake_set(key, value, ex=None):
        store[key] = value

    async def fake_get(key):
        return store.get(key)

    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.set = fake_set
    mock.get = fake_get
    return mock, store


def test_job_state_status_transitions():
    import time
    state = JobState(
        job_id="test-123",
        status=JobStatusEnum.queued,
        created_at=time.time(),
        updated_at=time.time(),
    )
    assert state.status == JobStatusEnum.queued
    state.status = JobStatusEnum.processing
    assert state.status == JobStatusEnum.processing
    state.status = JobStatusEnum.done
    assert state.status == JobStatusEnum.done


def test_job_state_serializes_with_result():
    import time
    result = GenerateResponse(
        cover_letter="Dear Hiring Manager...",
        resume_tweaks=[],
        match_score=75,
        matched_skills=["python"],
        missing_skills=["docker"],
        backend_used="ollama",
        model_used="llama3.2:3b",
    )
    state = JobState(
        job_id="abc",
        status=JobStatusEnum.done,
        result=result,
        created_at=time.time(),
        updated_at=time.time(),
    )
    serialized = state.model_dump_json()
    restored = JobState.model_validate_json(serialized)
    assert restored.result.cover_letter == "Dear Hiring Manager..."
    assert restored.result.match_score == 75


# ── WebSocket manager tests ───────────────────────────────────

def test_ws_manager_tracks_connections():
    from app.kafka.websocket_manager import WebSocketManager
    manager = WebSocketManager()
    assert len(manager._connections) == 0


@pytest.mark.asyncio
async def test_ws_broadcast_to_no_connections_is_safe():
    from app.kafka.websocket_manager import WebSocketManager
    manager = WebSocketManager()
    # Should not raise even with no connections
    await manager.broadcast("nonexistent-job", {"status": "done"})