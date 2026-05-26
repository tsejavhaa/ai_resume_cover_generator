from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum
import uuid


class ToneEnum(str, Enum):
    professional = "professional"
    confident = "confident"
    concise = "concise"
    enthusiastic = "enthusiastic"


class BackendEnum(str, Enum):
    ollama = "ollama"
    deepseek = "deepseek"


class JobStatusEnum(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


# ── Request ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    job_description: str = Field(..., min_length=50)
    tone: ToneEnum = ToneEnum.professional
    backend: Optional[BackendEnum] = None
    cover_letter_length: Literal["short", "medium", "long"] = "medium"


# ── Kafka message ─────────────────────────────────────────────

class JobMessage(BaseModel):
    """Serialized payload published to Kafka topic."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resume_bytes_b64: str        # base64-encoded file bytes
    filename: str
    job_description: str
    tone: str
    backend: Optional[str]
    cover_letter_length: str


# ── Redis job state ───────────────────────────────────────────

class JobState(BaseModel):
    """Persisted in Redis for status polling and WebSocket push."""
    job_id: str
    status: JobStatusEnum = JobStatusEnum.queued
    result: Optional["GenerateResponse"] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


# ── Intermediate / internal ───────────────────────────────────

class ExtractedProfile(BaseModel):
    raw_text: str
    skills: list[str]
    job_titles: list[str]
    organizations: list[str]
    education: list[str]


class JobProfile(BaseModel):
    raw_text: str
    required_skills: list[str]
    preferred_skills: list[str]
    job_title: str
    company: str


# ── Response ─────────────────────────────────────────────────

class ResumeTweak(BaseModel):
    section: str
    original: str
    suggested: str
    reason: str


class GenerateResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    cover_letter: str
    resume_tweaks: list[ResumeTweak]
    match_score: int = Field(..., ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    backend_used: str
    model_used: str


# ── Async API responses ───────────────────────────────────────

class SubmitResponse(BaseModel):
    """Returned immediately when a job is queued."""
    job_id: str
    status: JobStatusEnum
    message: str
    poll_url: str
    websocket_url: str


class StatusResponse(BaseModel):
    """Returned by GET /status/{job_id}."""
    job_id: str
    status: JobStatusEnum
    result: Optional[GenerateResponse] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    code: str


JobState.model_rebuild()