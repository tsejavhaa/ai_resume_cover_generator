"""
Backend-aware prompt templates for cover letter and resume generation.

DeepSeek gets more structured prompts with chain-of-thought reasoning,
explicit format examples, and stricter constraints. Llama gets concise
direct prompts suited to smaller models.
"""

import re

# ── Helpers ───────────────────────────────────────────────────

def _normalize_role_name(name: str) -> str:
    """Clean up a detected role for use in prompts."""
    if not name:
        return "the target role"
    name = re.sub(r"\s+", " ", name).strip()
    # Cap at ~5 words
    parts = name.split()
    if len(parts) > 5:
        name = " ".join(parts[:5]) + " Professional"
    return name


# ── Cover Letter ──────────────────────────────────────────────

LENGTH_INSTRUCTIONS = {
    "short": "3 paragraphs maximum, under 200 words",
    "medium": "4 paragraphs, 250-350 words",
    "long": "5-6 paragraphs, 400-500 words with specific examples",
}

COVER_LETTER_SYSTEM_LLAMA = """\
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

COVER_LETTER_SYSTEM_DEEPSEEK = """\
You are an expert career coach and professional writer. Write a tailored cover letter.

## OUTPUT RULES (strict)
1. Write in first person from the candidate's perspective
2. NO generic filler: avoid "I am excited to apply", "I am a team player", "I am writing to express my interest"
3. Mirror keywords from the job description naturally — do NOT just list them, weave them into accomplishments
4. Reference specific resume achievements with numbers/metrics where possible
5. Tone: {tone}
6. Length: {length_instruction}
7. Output ONLY the letter text — NO subject line, NO salutation "Dear Hiring Manager", NO signature block
8. NO commentary, NO explanations, NO "here is your cover letter" preface

## EXAMPLE OUTPUT
I led the migration of 12 microservices from on-prem to AWS ECS, reducing infrastructure costs by 35% while improving deployment frequency from weekly to daily. At my previous role, I built a real-time recommendation engine that increased user engagement by 22% over six months.
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

COVER_LETTER_USER_DEEPSEEK = """\
## RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## CONTEXT
- Candidate's key skills: {candidate_skills}
- Required skills for this role: {required_skills}
- Role title: {job_title}
- Company: {company}

## TASK
Write a tailored cover letter. Before the letter, think step by step:
1. Identify the 2-3 most impressive resume achievements that align with the job requirements
2. Identify which keywords from the JD are present in the resume (list them)
3. Draft a letter that opens with a strong achievement, connects it to the role, and closes with forward-looking statement

Then write the letter. Output ONLY the letter text.
"""


# ── Resume Tweaks ─────────────────────────────────────────────

RESUME_TWEAKS_SYSTEM_LLAMA = """\
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

RESUME_TWEAKS_SYSTEM_DEEPSEEK = """\
You are an expert ATS optimization specialist. Analyze the resume against the job description and suggest improvements.

## OUTPUT FORMAT (strict JSON — no markdown, no commentary)
```json
[
  {{
    "section": "Experience",
    "original": "Built ML models",
    "suggested": "Designed and deployed ML models serving 500K+ predictions daily, improving accuracy by 15%",
    "reason": "Adds quantifiable impact and aligns with JD requirement for production ML experience"
  }}
]
```

## RULES
- Exactly these keys: section, original, suggested, reason
- Maximum 5 suggestions
- original must be a real quote from the resume (never fabricate)
- suggested must be a realistic improvement (never fabricate experience)
- Include "section", "original", "suggested", and "reason" for EVERY entry
"""

RESUME_TWEAKS_USER = """\
## RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## ANALYSIS
- Skills the candidate has: {candidate_skills}
- Skills missing from the resume (listed in the JD): {missing_skills}
- Skills expected for this role but missing: {role_suggested_skills}
- Implicit role detected: {detected_role}

## INSTRUCTIONS
1. Suggest improvements to add missing skills where possible
2. For skills expected for the role (like HuggingFace, scikit-learn for ML roles),
   suggest that the candidate add them even if not explicitly in the JD —
   modern roles in this field typically require these at a beginner level
3. Explain WHY each suggested skill is important for this specific role
4. Do NOT suggest false experience — recommend learning resources or projects instead

Suggest specific, actionable resume improvements to better match this job.
Return a JSON array of objects with keys: section, original, suggested, reason.
"""

RESUME_TWEAKS_USER_DEEPSEEK = """\
## RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## ANALYSIS
- Candidate's skills: {candidate_skills}
- Missing from resume (in JD): {missing_skills}
- Role-expected skills missing: {role_suggested_skills}
- Inferred role: {detected_role}

## REASONING (do this internally, then output JSON)
1. Compare the JD requirements vs resume content — what key skills/experiences are missing?
2. For each missing skill, assess: can it be truthfully added to an existing job's bullet points?
3. For role-expected skills (e.g. HuggingFace for ML roles), suggest adding them even if not in JD
4. Choose the 3-5 most impactful changes that would improve ATS match rate

## OUTPUT
Return ONLY a valid JSON array with objects: section, original, suggested, reason.
"""


# ── Improved Resume ───────────────────────────────────────────

IMPROVED_RESUME_SYSTEM_LLAMA = """\
You are an expert resume writer and career coach specializing in ATS-optimized resumes.
Your task is to improve the candidate's resume to better match the job description.

Rules:
- Add missing skills from the job description into the resume naturally
- For each new skill added, create a beginner or pre-intermediate level experience bullet
  under the most relevant existing job (or a new "Projects & Learning" section)
- The added experience should be realistic for a beginner — mention tutorials, courses,
  small projects, or initial exposure — NOT production-level expertise
- NEVER claim advanced expertise the candidate doesn't have
- Keep all existing truthful experience intact
- Preserve the resume's original structure (Summary, Skills, Experience, Education)
- Output ONLY the improved resume text, no preamble or commentary
"""

IMPROVED_RESUME_SYSTEM_DEEPSEEK = """\
You are an expert resume writer and ATS optimization specialist.

## TASK
Improve the candidate's resume to better match the job description.

## APPROACH (reason through these steps before writing)
1. Parse the JD's required skills and responsibilities
2. Compare against the resume — identify gaps (skills in JD but missing from resume)
3. For each gap: determine which existing job section is the best place to add a beginner-level bullet
4. If no suitable section exists, add a "Projects & Continuing Education" section
5. Write the improved resume keeping the original structure intact

## CRITICAL RULES
- Added experience bullets must describe BEGINNER-level work: tutorials, courses, small projects, initial exposure. NEVER claim production experience.
- Use phrases like: "Completed an online course in…", "Built a proof-of-concept using…", "Gained foundational knowledge of…"
- Preserve every truthful achievement from the original resume — only ADD, never remove or inflate
- Keep the same section headers and overall structure
- Output ONLY the improved resume text — no reasoning, no commentary, no notes
"""

IMPROVED_RESUME_USER = """\
## ORIGINAL RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## ANALYSIS
- Candidate's existing skills: {candidate_skills}
- Skills missing from resume (listed in JD): {missing_skills}
- Skills expected for this role but missing: {role_suggested_skills}
- Inferred role: {detected_role}

## INSTRUCTIONS
1. Add the missing skills to the Skills section
2. For each NEW skill, add a brief experience bullet under the most relevant
   previous job in the Experience section. Each bullet should describe a
   beginner/pre-intermediate level task such as:
   - "Completed an online course/certification in [skill]"
   - "Built a small proof-of-concept project using [skill]"
   - "Gained foundational knowledge of [skill] through self-study"
   - "Applied [skill] in a supervised lab/tutorial environment"
3. Do NOT exaggerate the level of proficiency — use phrases like
   "beginner-level experience with", "familiarity with", "introductory knowledge of"
4. If the resume has no relevant job section for a skill, add a
   "Projects & Continuing Education" section at the bottom

Write the complete improved resume. Use the exact same section headers
as the original resume, just with improved content.
"""

IMPROVED_RESUME_USER_DEEPSEEK = """\
## ORIGINAL RESUME
{resume_text}

## JOB DESCRIPTION
{job_description}

## GAP ANALYSIS
- Candidate's skills: {candidate_skills}
- Skills in JD but missing from resume: {missing_skills}
- Role-expected skills not in resume: {role_suggested_skills}
- Inferred role: {detected_role}

## STEP-BY-STEP INSTRUCTIONS
1. First, list the exact section headers of the original resume (e.g. Summary, Skills, Experience, Education)
2. For EACH missing/role-expected skill, decide where to insert it:
   a. Add to the Skills section as a new line item
   b. Add a beginner-level experience bullet under the most relevant job
   c. If no job fits, create a "Projects & Continuing Education" section
3. For EVERY new skill bullet, use realistic beginner phrasing — never claim advanced expertise

## REALISTIC BEGINNER PHRASING (use these exact patterns)
- "Completed an online course in [Skill], building a [small project] to demonstrate proficiency"
- "Gained introductory knowledge of [Skill] through self-paced tutorials and hands-on practice"
- "Developed foundational familiarity with [Skill] in a supervised learning environment"
- "Applied [Skill] to a proof-of-concept project exploring [relevant use case]"

## OUTPUT REQUIREMENTS
- Keep ALL original section headers and their order
- Preserve every existing truthful bullet point
- Only add new content — never modify existing truthful claims to exaggerate
- Output ONLY the complete improved resume text, no analysis, no reasoning, no commentary
"""


# ── Prompt Builders ───────────────────────────────────────────

def build_cover_letter_messages(
    resume_text: str,
    job_description: str,
    candidate_skills: list[str],
    required_skills: list[str],
    job_title: str,
    company: str,
    tone: str,
    length: str,
    backend: str | None = None,
) -> list[dict]:
    is_deepseek = backend == "deepseek"
    system = (COVER_LETTER_SYSTEM_DEEPSEEK if is_deepseek else COVER_LETTER_SYSTEM_LLAMA).format(
        tone=tone,
        length_instruction=LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["medium"]),
    )
    user_tpl = COVER_LETTER_USER_DEEPSEEK if is_deepseek else COVER_LETTER_USER
    user = user_tpl.format(
        resume_text=resume_text[:3000],
        job_description=job_description[:2000],
        candidate_skills=", ".join(candidate_skills[:20]),
        required_skills=", ".join(required_skills[:15]),
        job_title=job_title,
        company=company,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_improve_resume_messages(
    resume_text: str,
    job_description: str,
    candidate_skills: list[str],
    missing_skills: list[str],
    role_suggested_skills: list[str] | None = None,
    detected_role: str = "",
    backend: str | None = None,
) -> list[dict]:
    is_deepseek = backend == "deepseek"
    system = IMPROVED_RESUME_SYSTEM_DEEPSEEK if is_deepseek else IMPROVED_RESUME_SYSTEM_LLAMA
    user_tpl = IMPROVED_RESUME_USER_DEEPSEEK if is_deepseek else IMPROVED_RESUME_USER
    user = user_tpl.format(
        resume_text=resume_text[:4000],
        job_description=job_description[:2000],
        candidate_skills=", ".join(candidate_skills[:20]),
        missing_skills=", ".join(missing_skills[:15]),
        role_suggested_skills=", ".join(role_suggested_skills[:15]) if role_suggested_skills else "none",
        detected_role=_normalize_role_name(detected_role),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_tweaks_messages(
    resume_text: str,
    job_description: str,
    candidate_skills: list[str],
    missing_skills: list[str],
    role_suggested_skills: list[str] | None = None,
    detected_role: str = "",
    backend: str | None = None,
) -> list[dict]:
    is_deepseek = backend == "deepseek"
    system = RESUME_TWEAKS_SYSTEM_DEEPSEEK if is_deepseek else RESUME_TWEAKS_SYSTEM_LLAMA
    user_tpl = RESUME_TWEAKS_USER_DEEPSEEK if is_deepseek else RESUME_TWEAKS_USER
    user = user_tpl.format(
        resume_text=resume_text[:3000],
        job_description=job_description[:2000],
        candidate_skills=", ".join(candidate_skills[:20]),
        missing_skills=", ".join(missing_skills[:15]),
        role_suggested_skills=", ".join(role_suggested_skills[:15]) if role_suggested_skills else "none",
        detected_role=_normalize_role_name(detected_role),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]