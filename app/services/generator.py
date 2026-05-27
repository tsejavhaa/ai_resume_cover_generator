"""
Generator service — the main orchestrator for Phase 1.

Pipeline:
   1. Parse resume file → raw text
   2. NLP extraction → structured profiles (resume + JD)
   3. Role-based skill inference + match scoring
   4. Build prompts → call LLM → cover letter
   5. Build prompts → call LLM → resume tweaks (JSON)
   6. Save history + return GenerateResponse
"""
import json
import re
import time
from loguru import logger

from app.models.schemas import (
    GenerateRequest,
    GenerateResponse,
    ResumeTweak,
)
from app.services.parser import extract_text
from app.services.nlp_extractor import (
    extract_resume_profile,
    extract_job_profile,
    compute_match_score,
    infer_role_skills,
)
from app.prompts.templates import build_cover_letter_messages, build_tweaks_messages
from app.services.llm_backends import get_backend
from app.services.history_service import save_history


async def generate(
    request: GenerateRequest,
    file_bytes: bytes,
    filename: str,
) -> GenerateResponse:
    """Full generation pipeline. Returns a GenerateResponse."""
    start_time = time.time()

    # ── Step 1: Parse document ───────────────────────────────
    logger.info(f"[1/6] Parsing resume: {filename}")
    resume_text = extract_text(file_bytes, filename)
    if not resume_text or len(resume_text) < 100:
        raise ValueError("Could not extract meaningful text from the uploaded file.")

    # ── Step 2: NLP extraction ───────────────────────────────
    logger.info("[2/6] Running NLP extraction")
    resume_profile = extract_resume_profile(resume_text)
    job_profile = extract_job_profile(request.job_description)
    logger.debug(f"Resume skills: {resume_profile.skills}")
    logger.debug(f"JD required skills: {job_profile.required_skills}")

    # ── Step 3a: Role-based skill gap analysis ────────────────
    logger.info("[3a/6] Inferring role-specific expected skills")
    role_expected = infer_role_skills(job_profile.job_title, job_profile.raw_text)
    resume_skill_set = set(s.lower() for s in resume_profile.skills)
    role_suggested = sorted(set(role_expected) - resume_skill_set)
    logger.info(
        f"Inferred role: {job_profile.job_title} | "
        f"expected skills: {role_expected} | "
        f"missing for role: {role_suggested}"
    )

    # ── Step 3b: Match scoring ────────────────────────────────
    logger.info("[3b/6] Computing match score")
    match_score, matched_skills, missing_skills = compute_match_score(
        resume_profile, job_profile
    )
    logger.info(f"Match score: {match_score}% | matched={matched_skills} | missing={missing_skills}")

    # ── Step 4: Select backend ───────────────────────────────
    backend_name = request.backend.value if request.backend else None
    backend = get_backend(override=backend_name)
    is_deepseek = backend.name == "deepseek"

    # ── Step 4: Generate cover letter ───────────────────────
    logger.info(f"[4/6] Generating cover letter via {backend.name}/{backend.model}")
    cl_messages = build_cover_letter_messages(
        resume_text=resume_profile.raw_text,
        job_description=job_profile.raw_text,
        candidate_skills=resume_profile.skills,
        required_skills=job_profile.required_skills,
        job_title=job_profile.job_title,
        company=job_profile.company,
        tone=request.tone.value,
        length=request.cover_letter_length,
        backend=backend.name,
    )
    cover_letter = await backend.chat(
        cl_messages,
        temperature=0.5 if is_deepseek else 0.75,
        max_tokens=1200 if is_deepseek else 800,
    )

    # ── Step 5: Generate resume tweaks ──────────────────────
    logger.info(f"[5/6] Generating resume tweaks via {backend.name}/{backend.model}")
    tweaks_messages = build_tweaks_messages(
        resume_text=resume_profile.raw_text,
        job_description=job_profile.raw_text,
        candidate_skills=resume_profile.skills,
        missing_skills=missing_skills,
        role_suggested_skills=role_suggested,
        detected_role=job_profile.job_title,
        backend=backend.name,
    )
    tweaks_raw = await backend.chat(
        tweaks_messages,
        temperature=0.2 if is_deepseek else 0.3,
        max_tokens=1500 if is_deepseek else 1000,
    )
    tweaks = _parse_tweaks(tweaks_raw)

    # Truncate resume text preview to ~500 chars
    preview = resume_text[:500]
    if len(resume_text) > 500:
        preview += "\n… [truncated]"

    response = GenerateResponse(
        cover_letter=cover_letter.strip(),
        resume_tweaks=tweaks,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        role_suggested_skills=role_suggested,
        resume_text_preview=preview,
        backend_used=backend.name,
        model_used=backend.model,
    )

    logger.info("[6/6] Saving to history")
    computation_time_ms = int((time.time() - start_time) * 1000)
    await save_history(
        entry_type="cover_letter",
        filename=filename,
        resume_text_preview=preview,
        job_description=job_profile.raw_text,
        backend_used=backend.name,
        model_used=backend.model,
        computation_time_ms=computation_time_ms,
        generate_response=response,
    )

    return response


def _parse_tweaks(raw: str) -> list[ResumeTweak]:
    """
    Safely parse the LLM's JSON array of resume tweaks.
    Strips markdown fences if present and validates structure.
    """
    # Strip ```json ... ``` if the model wrapped the output
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    # Find the first [ ... ] array in the output
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in tweaks response; returning empty list")
        return []
    try:
        data = json.loads(match.group())
        tweaks = []
        for item in data[:5]:  # cap at 5
            try:
                tweaks.append(
                    ResumeTweak(
                        section=item.get("section", "General"),
                        original=item.get("original", ""),
                        suggested=item.get("suggested", ""),
                        reason=item.get("reason", ""),
                    )
                )
            except Exception as e:
                logger.warning(f"Skipping malformed tweak item: {e}")
        return tweaks
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tweaks JSON: {e}\nRaw:\n{raw[:500]}")
        return []