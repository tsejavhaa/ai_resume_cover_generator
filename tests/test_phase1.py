"""
Unit tests for Phase 1 — NLP extractor and document parser.
Run with: pytest tests/ -v
"""
import pytest
from app.services.nlp_extractor import (
    _extract_skills_from_text,
    _extract_job_titles,
    _extract_education,
    compute_match_score,
    extract_resume_profile,
    extract_job_profile,
)
from app.services.parser import _clean, _parse_text


# ── Parser tests ─────────────────────────────────────────────

def test_clean_collapses_blank_lines():
    raw = "Line 1\n\n\n\n\nLine 2"
    assert _clean(raw) == "Line 1\n\n\nLine 2"


def test_parse_text_utf8():
    content = "Software Engineer\nPython, Docker, AWS"
    result = _parse_text(content.encode("utf-8"))
    assert "Software Engineer" in result
    assert "Python" in result


# ── Skill extraction tests ───────────────────────────────────

def test_skill_extraction_basic():
    text = "Experienced in Python, Docker, and Kubernetes for microservices deployment."
    skills = _extract_skills_from_text(text)
    assert "python" in skills
    assert "docker" in skills
    assert "kubernetes" in skills


def test_skill_extraction_no_false_positives():
    text = "I enjoy hiking and cooking on weekends."
    skills = _extract_skills_from_text(text)
    assert skills == []


def test_job_title_extraction():
    text = "Senior Software Engineer with 5 years of experience."
    titles = _extract_job_titles(text)
    assert len(titles) > 0
    assert any("engineer" in t.lower() for t in titles)


def test_education_extraction():
    text = "Bachelor of Science in Computer Science from MIT, 2018."
    edu = _extract_education(text)
    assert len(edu) > 0
    assert any("bachelor" in e.lower() for e in edu)


# ── Match score tests ────────────────────────────────────────

def test_match_score_perfect():
    resume_profile = extract_resume_profile(
        "Python, Docker, Kubernetes, AWS, FastAPI developer"
    )
    job_profile = extract_job_profile(
        "Required: Python, Docker. Preferred: Kubernetes, AWS. Senior FastAPI engineer role."
    )
    score, matched, missing = compute_match_score(resume_profile, job_profile)
    assert score > 60
    assert "python" in matched or "docker" in matched


def test_match_score_no_overlap():
    resume_profile = extract_resume_profile("Java Spring Boot developer")
    job_profile = extract_job_profile(
        "Required: Python, FastAPI, Kubernetes. Must have Docker experience."
    )
    score, matched, missing = compute_match_score(resume_profile, job_profile)
    assert score < 50
    assert len(missing) > 0


def test_match_score_bounds():
    resume_profile = extract_resume_profile("")
    job_profile = extract_job_profile("")
    score, _, _ = compute_match_score(resume_profile, job_profile)
    assert 0 <= score <= 100


# ── Full profile extraction ──────────────────────────────────

SAMPLE_RESUME = """
Jane Smith
Senior Data Scientist | jane@email.com

EXPERIENCE
Data Scientist — TechCorp (2020-2024)
- Built ML pipelines using Python, PyTorch, and Airflow
- Deployed models on AWS using Docker and Kubernetes

EDUCATION
Master of Science in Computer Science, Stanford University, 2020

SKILLS
Python, PyTorch, TensorFlow, SQL, Docker, Kubernetes, AWS, Airflow
"""

SAMPLE_JD = """
We are looking for a Senior ML Engineer.

Required: Python, PyTorch, Docker, Kubernetes
Preferred: TensorFlow, Airflow, AWS experience

You will build and maintain machine learning pipelines.
"""


def test_full_resume_profile():
    profile = extract_resume_profile(SAMPLE_RESUME)
    assert "python" in profile.skills
    assert "pytorch" in profile.skills


def test_full_job_profile():
    profile = extract_job_profile(SAMPLE_JD)
    assert "python" in profile.required_skills
    assert "pytorch" in profile.required_skills


def test_full_pipeline_score():
    resume = extract_resume_profile(SAMPLE_RESUME)
    job = extract_job_profile(SAMPLE_JD)
    score, matched, missing = compute_match_score(resume, job)
    assert score >= 60, f"Expected high match, got {score}%"
    assert "python" in matched