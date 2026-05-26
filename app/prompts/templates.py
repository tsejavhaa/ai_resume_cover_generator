"""
Versioned prompt templates for cover letter and resume tweak generation.
Keeping prompts in one place makes iteration fast without touching logic.
"""

# ── Cover Letter ──────────────────────────────────────────────

COVER_LETTER_SYSTEM = """\
You are an expert career coach and professional writer specializing in tailored job applications.
Your task is to write a compelling, personalized cover letter.

Rules:
- Write in first person
- Do NOT use generic filler phrases like "I am excited to apply" or "I am a team player"
- Mirror language and keywords from the job description naturally
- Reference specific skills and achievements from the resume
- Tone: {tone}
- Length: {length_instruction}
- Output ONLY the cover letter text, no preamble or commentary
"""

COVER_LETTER_USER = """\
## RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## CONTEXT
- Candidate's key skills: {candidate_skills}
- Required skills for this role: {required_skills}
- Role title: {job_title}
- Company: {company}

Write a tailored cover letter for this candidate applying to this specific role.
"""

LENGTH_INSTRUCTIONS = {
    "short": "3 paragraphs maximum, under 200 words",
    "medium": "4 paragraphs, 250-350 words",
    "long": "5-6 paragraphs, 400-500 words with specific examples",
}

# ── Resume Tweaks ─────────────────────────────────────────────

RESUME_TWEAKS_SYSTEM = """\
You are an expert ATS (Applicant Tracking System) optimization specialist and career coach.
Analyze the resume against the job description and suggest targeted improvements.

Rules:
- Return a valid JSON array only, no markdown fences, no commentary
- Each object has exactly these keys: section, original, suggested, reason
- Maximum 5 suggestions
- Focus on: missing keywords, weak bullet points, quantifiable achievements
- Do NOT suggest adding false information
- Output ONLY the JSON array
"""

RESUME_TWEAKS_USER = """\
## RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## ANALYSIS
- Skills the candidate has: {candidate_skills}
- Skills missing from the resume: {missing_skills}

Suggest specific, actionable resume improvements to better match this job.
Return a JSON array of objects with keys: section, original, suggested, reason.
"""


def build_cover_letter_messages(
    resume_text: str,
    job_description: str,
    candidate_skills: list[str],
    required_skills: list[str],
    job_title: str,
    company: str,
    tone: str,
    length: str,
) -> list[dict]:
    system = COVER_LETTER_SYSTEM.format(
        tone=tone,
        length_instruction=LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["medium"]),
    )
    user = COVER_LETTER_USER.format(
        resume_text=resume_text[:3000],
        job_description=job_description[:2000],
        candidate_skills=", ".join(candidate_skills[:20]),
        required_skills=", ".join(required_skills[:15]),
        job_title=job_title,
        company=company,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_tweaks_messages(
    resume_text: str,
    job_description: str,
    candidate_skills: list[str],
    missing_skills: list[str],
) -> list[dict]:
    user = RESUME_TWEAKS_USER.format(
        resume_text=resume_text[:3000],
        job_description=job_description[:2000],
        candidate_skills=", ".join(candidate_skills[:20]),
        missing_skills=", ".join(missing_skills[:15]),
    )
    return [
        {"role": "system", "content": RESUME_TWEAKS_SYSTEM},
        {"role": "user", "content": user},
    ]