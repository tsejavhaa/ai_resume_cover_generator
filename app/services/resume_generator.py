"""
Resume improvement generator — creates an enhanced resume.

Pipeline:
  1. Parse resume file → raw text
  2. NLP extraction → structured profiles
  3. Role-based skill inference + match scoring + gap analysis
  4. Build prompt → call LLM → improved resume text
  5. Parse LLM output into structured sections
  6. Save history + return ImproveResumeResponse
"""
import json
import re
import time
from loguru import logger

from app.models.schemas import (
    ImproveResumeRequest, ImproveResumeResponse, ImprovedResumeSection,
)
from app.services.parser import extract_text
from app.services.nlp_extractor import (
    extract_resume_profile, extract_job_profile,
    compute_match_score, infer_role_skills,
)
from app.prompts.templates import build_improve_resume_messages
from app.services.llm_backends import get_backend
from app.services.history_service import save_history


async def improve_resume(
    request: ImproveResumeRequest,
    file_bytes: bytes,
    filename: str,
) -> ImproveResumeResponse:
    start_time = time.time()

    logger.info(f"[1/5] Parsing resume: {filename}")
    resume_text = extract_text(file_bytes, filename)
    if not resume_text or len(resume_text) < 100:
        raise ValueError("Could not extract meaningful text from the uploaded file.")

    logger.info("[2/5] Running NLP extraction")
    resume_profile = extract_resume_profile(resume_text)
    job_profile = extract_job_profile(request.job_description)

    logger.info("[3a/5] Inferring role-specific expected skills")
    role_expected = infer_role_skills(job_profile.job_title, job_profile.raw_text)
    resume_skill_set = set(s.lower() for s in resume_profile.skills)
    role_suggested = sorted(set(role_expected) - resume_skill_set)

    logger.info("[3b/5] Computing match score")
    match_score, matched_skills, missing_skills = compute_match_score(
        resume_profile, job_profile
    )

    all_new_skills = set(missing_skills) | set(role_suggested)
    new_skills_added = sorted(all_new_skills - resume_skill_set)

    backend_name = request.backend.value if request.backend else None
    backend = get_backend(override=backend_name)
    is_deepseek = backend.name == "deepseek"

    logger.info(f"[4/5] Generating improved resume via {backend.name}/{backend.model}")
    messages = build_improve_resume_messages(
        resume_text=resume_profile.raw_text,
        job_description=job_profile.raw_text,
        candidate_skills=resume_profile.skills,
        missing_skills=missing_skills,
        role_suggested_skills=role_suggested,
        detected_role=job_profile.job_title,
        backend=backend.name,
    )
    improved_text = await backend.chat(
        messages,
        temperature=0.3 if is_deepseek else 0.4,
        max_tokens=3000 if is_deepseek else 2000,
    )

    logger.info("[5/5] Extracting improved sections")
    sections = _extract_sections(resume_text, improved_text)
    if not sections and new_skills_added:
        sections.append(ImprovedResumeSection(
            section="Skills & Experience",
            original="Resume rewritten to include missing skills",
            improved=improved_text[:500],
            reason=f"Added {len(new_skills_added)} new skills at beginner level",
        ))

    preview = resume_text[:500]
    if len(resume_text) > 500:
        preview += "\n… [truncated]"

    response = ImproveResumeResponse(
        improved_resume_text=improved_text.strip(),
        improved_sections=sections,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        new_skills_added=new_skills_added,
        role_suggested_skills=role_suggested,
        resume_text_preview=preview,
        backend_used=backend.name,
        model_used=backend.model,
    )

    computation_time_ms = int((time.time() - start_time) * 1000)

    await save_history(
        entry_type="resume",
        filename=filename,
        resume_text_preview=preview,
        job_description=job_profile.raw_text,
        backend_used=backend.name,
        model_used=backend.model,
        computation_time_ms=computation_time_ms,
        improve_response=response,
    )

    return response


def _extract_sections(original: str, improved: str) -> list[ImprovedResumeSection]:
    """
    Attempt to identify what changed between original and improved resume
    by comparing section by section.
    """
    sections = []
    original_lines = original.splitlines()
    improved_lines = improved.splitlines()

    orig_by_section = _split_into_sections(original_lines)
    improved_by_section = _split_into_sections(improved_lines)

    for section_name, orig_content in orig_by_section.items():
        imp_content = improved_by_section.get(section_name, "")
        if orig_content.strip() != imp_content.strip() and imp_content.strip():
            sections.append(ImprovedResumeSection(
                section=section_name,
                original=orig_content.strip()[:500],
                improved=imp_content.strip()[:500],
                reason=f"Enhanced {section_name} section to better match job requirements",
            ))

    new_sections = set(improved_by_section.keys()) - set(orig_by_section.keys())
    for section_name in new_sections:
        imp_content = improved_by_section[section_name]
        if imp_content.strip():
            sections.append(ImprovedResumeSection(
                section=section_name,
                original="(not present in original)",
                improved=imp_content.strip()[:500],
                reason=f"Added new section: {section_name}",
            ))

    return sections


def _split_into_sections(lines: list[str]) -> dict[str, str]:
    """
    Split resume text into sections based on common section headers.
    """
    section_headers = re.compile(
        r"^(summary|experience|work experience|employment|education|skills"
        r"|projects|publications|certifications|languages|interests"
        r"|projects & continuing education|additional)",
        re.IGNORECASE,
    )
    sections: dict[str, list[str]] = {}
    current_section = "preamble"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = section_headers.match(stripped)
        if match:
            current_section = match.group(1).lower()
            if current_section not in sections:
                sections[current_section] = []
        if current_section not in sections:
            sections[current_section] = []
        sections[current_section].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}
