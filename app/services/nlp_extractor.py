"""
NLP extraction layer.
Uses HuggingFace BERT-NER to pull named entities, then applies
keyword heuristics to classify skills, titles, orgs, and education.
"""
import re
from functools import lru_cache
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import ExtractedProfile, JobProfile

# ── Common tech skills vocabulary for keyword matching ───────
SKILL_KEYWORDS = {
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "fastapi", "django", "flask", "react", "vue", "angular", "node.js",
    "docker", "kubernetes", "terraform", "aws", "gcp", "azure", "ci/cd",
    "machine learning", "deep learning", "nlp", "data science", "pytorch",
    "tensorflow", "huggingface", "transformers", "kafka", "rabbitmq",
    "graphql", "rest", "grpc", "microservices", "agile", "scrum",
    "git", "linux", "bash", "spark", "hadoop", "airflow", "dbt",
}

EDUCATION_KEYWORDS = {
    "bachelor", "master", "phd", "b.s.", "m.s.", "b.a.", "m.a.",
    "degree", "university", "college", "institute", "school",
    "computer science", "engineering", "mathematics",
}

JOB_TITLE_PATTERNS = [
    r"\b(senior|junior|lead|principal|staff)?\s*(software|data|ml|ai|devops|backend|frontend|fullstack|full-stack)\s*(engineer|developer|scientist|analyst|architect)\b",
    r"\b(engineering|product|project|program)\s*manager\b",
    r"\b(vp|director|head)\s+of\s+\w+\b",
    r"\bcto\b|\bceo\b|\bcfo\b|\bcoo\b",
]

# ── Role → expected skills knowledge base ──────────────────────
# Maps normalized role key to a set of modern tools/technologies
# that professionals in that role are typically expected to know.
ROLE_SKILL_MAP: dict[str, set[str]] = {
    "machine learning": {
        "pytorch", "tensorflow", "huggingface", "transformers",
        "scikit-learn", "pandas", "numpy", "jupyter", "mlflow",
    },
    "data science": {
        "pandas", "numpy", "scikit-learn", "jupyter",
        "sql", "r", "tableau", "matplotlib", "seaborn",
    },
    "data engineer": {
        "spark", "airflow", "kafka", "sql", "python",
        "aws", "gcp", "azure", "hadoop", "snowflake", "bigquery", "dbt",
    },
    "devops": {
        "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "ci/cd", "aws", "gcp", "azure",
        "prometheus", "grafana", "linux",
    },
    "backend": {
        "fastapi", "django", "flask", "sql", "redis",
        "kafka", "docker", "aws", "rest", "graphql", "grpc",
    },
    "frontend": {
        "react", "vue", "angular", "typescript", "css",
        "html", "nextjs", "tailwind", "sass",
    },
    "fullstack": {
        "react", "node.js", "typescript", "python",
        "sql", "docker", "aws", "graphql", "rest",
    },
    "ai engineer": {
        "pytorch", "tensorflow", "huggingface", "langchain",
        "vector database", "rag", "llm", "openai", "mlflow",
        "python", "docker",
    },
    "data analyst": {
        "sql", "python", "r", "tableau", "power bi",
        "excel", "pandas", "numpy",
    },
    "software engineer": {
        "python", "java", "javascript", "typescript",
        "git", "docker", "sql", "rest", "ci/cd",
        "aws", "gcp", "agile",
    },
}

# Role detection patterns — mapped to ROLE_SKILL_MAP keys
ROLE_DETECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(ml|machine learning|ai|artificial intelligence)\s*(engineer|scientist|specialist)\b", re.IGNORECASE), "machine learning"),
    (re.compile(r"\b(ai|ml)\s*(engineer|developer)\b", re.IGNORECASE), "ai engineer"),
    (re.compile(r"\bdata\s*scientist\b", re.IGNORECASE), "data science"),
    (re.compile(r"\bdata\s*engineer\b", re.IGNORECASE), "data engineer"),
    (re.compile(r"\bdata\s*analyst\b", re.IGNORECASE), "data analyst"),
    (re.compile(r"\bdevops\s*(engineer|specialist)?\b", re.IGNORECASE), "devops"),
    (re.compile(r"\bsre\b|\bsite\s*reliability\s*engineer\b", re.IGNORECASE), "devops"),
    (re.compile(r"\bbackend\s*(engineer|developer)?\b", re.IGNORECASE), "backend"),
    (re.compile(r"\bfrontend\s*(engineer|developer)?\b", re.IGNORECASE), "frontend"),
    (re.compile(r"\bfull.?stack\s*(engineer|developer)?\b", re.IGNORECASE), "fullstack"),
    (re.compile(r"\bsoftware\s*(engineer|developer)\b", re.IGNORECASE), "software engineer"),
]


def infer_role_skills(job_title: str, job_description: str = "") -> list[str]:
    """
    Infer what skills are expected for the detected role.
    Checks the job title first, then falls back to scanning the JD.
    Returns a sorted list of recommended skill keywords.
    """
    combined = f"{job_title} {job_description}"
    matched_roles: set[str] = set()

    for pattern, role_key in ROLE_DETECTION_PATTERNS:
        if pattern.search(combined):
            matched_roles.add(role_key)

    # Collect all expected skills for matched roles
    expected: set[str] = set()
    for role in matched_roles:
        expected |= ROLE_SKILL_MAP.get(role, set())

    return sorted(expected)


REQUIRED_SIGNAL = re.compile(
    r"\b(required|must have|minimum|mandatory|you (have|bring|possess))\b",
    re.IGNORECASE,
)
PREFERRED_SIGNAL = re.compile(
    r"\b(preferred|nice to have|bonus|plus|ideally|familiarity)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_ner_pipeline():
    """Lazy-load the NER model once and cache it."""
    settings = get_settings()
    logger.info(f"Loading NER model: {settings.ner_model}")
    try:
        from transformers import pipeline
        ner = pipeline(
            "ner",
            model=settings.ner_model,
            aggregation_strategy="simple",
            device=-1,  # CPU; change to 0 for GPU
        )
        logger.info("NER model loaded successfully")
        return ner
    except Exception as e:
        logger.warning(f"NER model unavailable ({e}); falling back to keyword-only mode")
        return None


def _run_ner(text: str) -> dict[str, list[str]]:
    """Run NER and group results by entity type."""
    ner = _load_ner_pipeline()
    entities: dict[str, list[str]] = {"PER": [], "ORG": [], "LOC": [], "MISC": []}
    if ner is None:
        return entities
    try:
        results = ner(text[:2000])  # cap to avoid OOM on very long docs
        for ent in results:
            label = ent.get("entity_group", "MISC")
            word = ent.get("word", "").strip()
            if word and label in entities:
                entities[label].append(word)
    except Exception as e:
        logger.warning(f"NER inference error: {e}")
    return entities


def _extract_skills_from_text(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if re.search(r"\b" + re.escape(skill) + r"\b", lower):
            found.append(skill)
    return sorted(set(found))


def _extract_education(text: str) -> list[str]:
    lower = text.lower()
    lines = text.splitlines()
    edu = []
    for line in lines:
        if any(kw in line.lower() for kw in EDUCATION_KEYWORDS):
            cleaned = line.strip()
            if cleaned:
                edu.append(cleaned)
    return edu[:5]  # top 5 most relevant lines


def _extract_job_titles(text: str) -> list[str]:
    titles = []
    for pattern in JOB_TITLE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            title = " ".join(m).strip()
            if title:
                titles.append(title)
    return list(dict.fromkeys(titles))[:5]


# ── Public API ───────────────────────────────────────────────

def extract_resume_profile(text: str) -> ExtractedProfile:
    """Extract structured profile from resume text."""
    entities = _run_ner(text)
    return ExtractedProfile(
        raw_text=text,
        skills=_extract_skills_from_text(text),
        job_titles=_extract_job_titles(text),
        organizations=list(dict.fromkeys(entities["ORG"]))[:8],
        education=_extract_education(text),
    )


def extract_job_profile(text: str) -> JobProfile:
    """Extract structured requirements from the job description."""
    entities = _run_ner(text)
    all_skills = _extract_skills_from_text(text)

    # Split into required vs preferred based on surrounding context
    required, preferred = [], []
    sentences = re.split(r"[.\n]", text)
    for sentence in sentences:
        skills_in_sentence = [s for s in all_skills if s in sentence.lower()]
        if not skills_in_sentence:
            continue
        if REQUIRED_SIGNAL.search(sentence):
            required.extend(skills_in_sentence)
        elif PREFERRED_SIGNAL.search(sentence):
            preferred.extend(skills_in_sentence)
        else:
            required.extend(skills_in_sentence)

    # Fallback: all skills → required if not split
    if not required and not preferred:
        required = all_skills

    # Extract company name (first ORG entity) and job title
    orgs = list(dict.fromkeys(entities["ORG"]))
    titles = _extract_job_titles(text)

    return JobProfile(
        raw_text=text,
        required_skills=list(dict.fromkeys(required)),
        preferred_skills=list(dict.fromkeys(preferred)),
        job_title=titles[0] if titles else "the advertised role",
        company=orgs[0] if orgs else "the company",
    )


def compute_match_score(
    resume: ExtractedProfile, job: JobProfile
) -> tuple[int, list[str], list[str]]:
    """
    Compute a 0-100 keyword match score.
    Returns (score, matched_skills, missing_skills).
    """
    resume_skills = set(s.lower() for s in resume.skills)
    required = set(s.lower() for s in job.required_skills)
    preferred = set(s.lower() for s in job.preferred_skills)
    all_jd_skills = required | preferred

    if not all_jd_skills:
        return 50, [], []

    matched = resume_skills & all_jd_skills
    missing = all_jd_skills - resume_skills

    # Weight required skills more heavily
    required_matched = resume_skills & required
    preferred_matched = resume_skills & preferred
    req_score = (len(required_matched) / max(len(required), 1)) * 70
    pref_score = (len(preferred_matched) / max(len(preferred), 1)) * 30
    score = int(min(req_score + pref_score, 100))

    return score, sorted(matched), sorted(missing)