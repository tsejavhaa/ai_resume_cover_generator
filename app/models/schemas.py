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
    role_suggested_skills: list[str] = Field(
        default_factory=list,
        description="Skills expected for the inferred role but missing from resume",
    )
    resume_text_preview: str = Field(
        default="",
        description="First ~500 chars of parsed resume text for preview",
        max_length=1000,
    )
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


class PreviewResponse(BaseModel):
    """Returned by POST /api/v1/preview."""
    filename: str
    resume_text_preview: str
    skills: list[str]
    job_titles: list[str]
    organizations: list[str]
    education: list[str]
    inferred_role: str
    role_expected_skills: list[str]


class ImprovedResumeSection(BaseModel):
    section: str
    original: str
    improved: str
    reason: str


class ImproveResumeRequest(BaseModel):
    job_description: str = Field(..., min_length=50)
    tone: ToneEnum = ToneEnum.professional
    backend: Optional[BackendEnum] = None


class ImproveResumeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    improved_resume_text: str
    improved_sections: list[ImprovedResumeSection]
    match_score: int = Field(..., ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    new_skills_added: list[str]
    role_suggested_skills: list[str] = Field(
        default_factory=list,
        description="Skills expected for the inferred role but missing from resume",
    )
    resume_text_preview: str = Field(
        default="",
        description="Original resume preview (~500 chars)",
        max_length=1000,
    )
    backend_used: str
    model_used: str


class JobHistoryEntry(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    type: Literal["cover_letter", "resume"]
    created_at: float
    filename: str
    resume_text_preview: str
    job_description: str
    # Cover letter results
    cover_letter: Optional[str] = None
    resume_tweaks: list[ResumeTweak] = []
    # Resume improvement results
    improved_resume_text: Optional[str] = None
    improved_sections: list[ImprovedResumeSection] = []
    new_skills_added: list[str] = []
    # Shared
    match_score: int = 0
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    role_suggested_skills: list[str] = []
    backend_used: str
    model_used: str
    computation_time_ms: int = 0


class JobHistoryListItem(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    type: str
    created_at: float
    filename: str
    match_score: int
    backend_used: str
    model_used: str


class JobHistoryListResponse(BaseModel):
    entries: list[JobHistoryListItem]


class ErrorResponse(BaseModel):
    detail: str
    code: str


JobState.model_rebuild()