"""
FastAPI routes — Phase 1 + Phase 2.

Phase 1 (sync):
  POST /api/v1/generate        → wait for result inline

Phase 2 (async):
  POST /api/v1/submit          → enqueue job, return job_id immediately
  GET  /api/v1/status/{job_id} → poll for result
  WS   /api/v1/ws/{job_id}     → real-time push when job completes

GET  /api/v1/health            → backend availability
"""
import traceback
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, status
from loguru import logger

from app.models.schemas import (
    GenerateRequest, GenerateResponse,
    JobMessage, SubmitResponse, StatusResponse,
    ToneEnum, BackendEnum, ErrorResponse, JobStatusEnum,
)
from app.services.generator import generate
from app.services.llm_backends import get_backend
from app.services.storage.redis_store import create_job, get_job
from app.kafka.producer import publish_job, encode_file
from app.kafka.websocket_manager import ws_manager

router = APIRouter(prefix="/api/v1", tags=["generate"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/octet-stream",
}


def _validate_upload(resume: UploadFile, file_bytes: bytes) -> None:
    if resume.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {resume.content_type}")
    if len(file_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum 5 MB.")


# ── Phase 1: Sync endpoint ────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse, summary="[Sync] Generate and wait")
async def generate_sync(
    resume: UploadFile = File(...),
    job_description: str = Form(..., min_length=50),
    tone: ToneEnum = Form(default=ToneEnum.professional),
    backend: BackendEnum | None = Form(default=None),
    cover_letter_length: str = Form(default="medium"),
):
    file_bytes = await resume.read()
    _validate_upload(resume, file_bytes)
    if cover_letter_length not in ("short", "medium", "long"):
        cover_letter_length = "medium"

    request = GenerateRequest(
        job_description=job_description, tone=tone,
        backend=backend, cover_letter_length=cover_letter_length,
    )
    logger.info(f"[sync] file={resume.filename} size={len(file_bytes)}B")
    try:
        return await generate(request, file_bytes, resume.filename or "resume.txt")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Sync generation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Generation failed: {type(e).__name__}: {e}")


# ── Phase 2: Async submit ─────────────────────────────────────

@router.post("/submit", response_model=SubmitResponse, summary="[Async] Enqueue job via Kafka")
async def submit_job(
    resume: UploadFile = File(...),
    job_description: str = Form(..., min_length=50),
    tone: ToneEnum = Form(default=ToneEnum.professional),
    backend: BackendEnum | None = Form(default=None),
    cover_letter_length: str = Form(default="medium"),
):
    file_bytes = await resume.read()
    _validate_upload(resume, file_bytes)
    if cover_letter_length not in ("short", "medium", "long"):
        cover_letter_length = "medium"

    # Build Kafka message
    message = JobMessage(
        resume_bytes_b64=encode_file(file_bytes),
        filename=resume.filename or "resume.txt",
        job_description=job_description,
        tone=tone.value,
        backend=backend.value if backend else None,
        cover_letter_length=cover_letter_length,
    )

    # Persist job state in Redis and publish to Kafka
    await create_job(message.job_id)
    await publish_job(message)

    logger.info(f"[async] Queued job {message.job_id} file={resume.filename}")

    return SubmitResponse(
        job_id=message.job_id,
        status=JobStatusEnum.queued,
        message="Job queued. Poll /status or connect to /ws for updates.",
        poll_url=f"/api/v1/status/{message.job_id}",
        websocket_url=f"/api/v1/ws/{message.job_id}",
    )


# ── Phase 2: Status polling ───────────────────────────────────

@router.get("/status/{job_id}", response_model=StatusResponse, summary="Poll job status")
async def job_status(job_id: str):
    state = await get_job(job_id)
    if state is None:
        raise HTTPException(404, f"Job '{job_id}' not found or expired.")
    return StatusResponse(
        job_id=state.job_id,
        status=state.status,
        result=state.result,
        error=state.error,
    )


# ── Phase 2: WebSocket push ───────────────────────────────────

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        # Send current state immediately on connect
        state = await get_job(job_id)
        if state:
            await websocket.send_json({
                "status": state.status.value,
                "job_id": job_id,
                **({"result": state.result.model_dump()} if state.result else {}),
                **({"error": state.error} if state.error else {}),
            })
        # Keep connection alive until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)


# ── Health check ──────────────────────────────────────────────

@router.get("/health", summary="Backend availability")
async def health():
    backends = {}
    for name in ("ollama", "deepseek"):
        try:
            b = get_backend(override=name)
            backends[name] = {"available": await b.is_available(), "model": b.model}
        except Exception as e:
            backends[name] = {"available": False, "error": str(e)}
    return {"status": "ok", "backends": backends}