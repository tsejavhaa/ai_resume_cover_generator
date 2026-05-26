"""
Generator service — the main orchestrator for Phase 1.

Pipeline:
  1. Parse resume file → raw text
  2. NLP extraction → structured profiles (resume + JD)
  3. Compute match score + gap analysis
  4. Build prompts → call LLM → cover letter
  5. Build prompts → call LLM → resume tweaks (JSON)
  6. Assemble and return GenerateResponse
"""
import json
import re
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
)
from app.prompts.templates import build_cover_letter_messages, build_tweaks_messages
from app.services.llm_backends import get_backend


async def generate(
    request: GenerateRequest,
    file_bytes: bytes,
    filename: str,
) -> GenerateResponse:
    """Full generation pipeline. Returns a GenerateResponse."""

    # ── Step 1: Parse document ───────────────────────────────
    logger.info(f"[1/5] Parsing resume: {filename}")
    resume_text = extract_text(file_bytes, filename)
    if not resume_text or len(resume_text) < 100:
        raise ValueError("Could not extract meaningful text from the uploaded file.")

    # ── Step 2: NLP extraction ───────────────────────────────
    logger.info("[2/5] Running NLP extraction")
    resume_profile = extract_resume_profile(resume_text)
    job_profile = extract_job_profile(request.job_description)
    logger.debug(f"Resume skills: {resume_profile.skills}")
    logger.debug(f"JD required skills: {job_profile.required_skills}")

    # ── Step 3: Match scoring ────────────────────────────────
    logger.info("[3/5] Computing match score")
    match_score, matched_skills, missing_skills = compute_match_score(
        resume_profile, job_profile
    )
    logger.info(f"Match score: {match_score}% | matched={matched_skills} | missing={missing_skills}")

    # ── Step 4: Select backend ───────────────────────────────
    backend_name = request.backend.value if request.backend else None
    backend = get_backend(override=backend_name)

    # ── Step 5: Generate cover letter ───────────────────────
    logger.info(f"[4/5] Generating cover letter via {backend.name}/{backend.model}")
    cl_messages = build_cover_letter_messages(
        resume_text=resume_profile.raw_text,
        job_description=job_profile.raw_text,
        candidate_skills=resume_profile.skills,
        required_skills=job_profile.required_skills,
        job_title=job_profile.job_title,
        company=job_profile.company,
        tone=request.tone.value,
        length=request.cover_letter_length,
    )
    cover_letter = await backend.chat(cl_messages, temperature=0.75, max_tokens=800)

    # ── Step 6: Generate resume tweaks ──────────────────────
    logger.info(f"[5/5] Generating resume tweaks via {backend.name}/{backend.model}")
    tweaks_messages = build_tweaks_messages(
        resume_text=resume_profile.raw_text,
        job_description=job_profile.raw_text,
        candidate_skills=resume_profile.skills,
        missing_skills=missing_skills,
    )
    tweaks_raw = await backend.chat(tweaks_messages, temperature=0.3, max_tokens=1000)
    tweaks = _parse_tweaks(tweaks_raw)

    return GenerateResponse(
        cover_letter=cover_letter.strip(),
        resume_tweaks=tweaks,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        backend_used=backend.name,
        model_used=backend.model,
    )


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