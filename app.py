import streamlit as st
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

import PyPDF2
import io
import re
import anthropic
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import datetime
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    APP_TITLE = "AI Resume Analyzer"
    VERSION = "2.0.0"
    SKILL_MATCH_WEIGHT = 0.4
    CONTENT_SIMILARITY_WEIGHT = 0.35
    KEYWORD_DENSITY_WEIGHT = 0.25
    EXCELLENT_SCORE = 80
    GOOD_SCORE = 60
    FAIR_SCORE = 40

# ============================================================================
# STOPWORDS
# ============================================================================

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'resume', 'cv', 'curriculum', 'vitae', 'page', 'phone', 'email', 'address',
    'linkedin', 'github', 'portfolio', 'objective', 'summary', 'experience',
    'education', 'skills', 'references', 'available', 'upon', 'request'
}

# ============================================================================
# SKILLS DATABASE
# ============================================================================

SKILLS_DB = {
    "Programming Languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby",
        "php", "swift", "kotlin", "go", "golang", "rust", "scala", "perl", "r",
        "matlab", "sql", "bash", "powershell", "shell"
    },
    "Web Technologies": {
        "html", "html5", "css", "css3", "sass", "scss", "bootstrap", "tailwind",
        "react", "reactjs", "angular", "vue", "vuejs", "nodejs", "node", "express",
        "django", "flask", "fastapi", "spring", "spring boot", "asp.net", ".net",
        "laravel", "rails", "ruby on rails", "rest", "restful", "api", "graphql",
        "webpack", "npm", "yarn", "jquery", "nextjs", "nuxt"
    },
    "Databases": {
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "oracle",
        "sql server", "redis", "elasticsearch", "cassandra", "dynamodb",
        "firebase", "mariadb", "neo4j", "couchdb"
    },
    "Cloud & DevOps": {
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
        "kubernetes", "k8s", "jenkins", "terraform", "ansible", "ci/cd", "git",
        "github", "gitlab", "bitbucket", "linux", "nginx", "apache", "heroku",
        "digitalocean", "cloudformation", "ec2", "s3", "lambda", "ecs", "eks"
    },
    "Data Science & ML": {
        "machine learning", "deep learning", "artificial intelligence", "ai",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn", "nlp", "natural language processing",
        "computer vision", "neural networks", "data analysis", "data science",
        "jupyter", "statistics", "regression", "classification", "clustering"
    },
    "Tools & Software": {
        "jira", "confluence", "slack", "trello", "asana", "figma", "sketch",
        "photoshop", "illustrator", "excel", "powerpoint", "word", "tableau",
        "power bi", "postman", "vs code", "intellij", "eclipse"
    },
    "Soft Skills": {
        "communication", "leadership", "teamwork", "problem solving",
        "analytical", "creativity", "adaptability", "time management",
        "project management", "agile", "scrum", "kanban"
    }
}

def get_all_skills() -> Set[str]:
    all_skills = set()
    for category_skills in SKILLS_DB.values():
        all_skills.update(category_skills)
    return all_skills

def get_skill_category(skill: str) -> Optional[str]:
    skill_lower = skill.lower()
    for category, skills in SKILLS_DB.items():
        if skill_lower in skills:
            return category
    return None

# ============================================================================
# TEXT PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return ' '.join(text_parts)
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> List[str]:
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1 and not w.isdigit()]

def extract_skills(text: str) -> Set[str]:
    if not text:
        return set()
    text_lower = text.lower()
    found_skills = set()
    all_skills = get_all_skills()
    for skill in all_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills

def categorize_skills(skills: Set[str]) -> Dict[str, List[str]]:
    categorized = {}
    for skill in skills:
        category = get_skill_category(skill)
        if category:
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(skill)
    for cat in categorized:
        categorized[cat].sort()
    return categorized

def get_keywords(text: str, top_n: int = 30) -> List[str]:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(top_n)]

# ============================================================================
# ANALYSIS RESULT
# ============================================================================

@dataclass
class AnalysisResult:
    overall_score: float
    skill_match_score: float
    content_similarity_score: float
    keyword_match_score: float
    matched_skills: Set[str]
    missing_skills: Set[str]
    resume_only_skills: Set[str]
    matched_keywords: List[str]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    skill_categories: Dict[str, List[str]]
    missing_categories: Dict[str, List[str]]

# ============================================================================
# ATS ENGINE
# ============================================================================

def calculate_similarity(text1: str, text2: str) -> float:
    try:
        t1 = clean_text(text1)
        t2 = clean_text(text2)
        if not t1 or not t2:
            return 0.0
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        matrix = vectorizer.fit_transform([t1, t2])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return sim * 100
    except:
        return 0.0

def analyze_resume(resume_text: str, job_description: str) -> AnalysisResult:
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra = resume_skills - jd_skills

    skill_score = (len(matched) / len(jd_skills) * 100) if jd_skills else 100.0
    content_score = calculate_similarity(resume_text, job_description)

    jd_keywords = get_keywords(job_description)
    resume_lower = resume_text.lower()
    matched_kw = [kw for kw in jd_keywords if kw in resume_lower]
    keyword_score = (len(matched_kw) / len(jd_keywords) * 100) if jd_keywords else 100.0

    overall = (
        skill_score * Config.SKILL_MATCH_WEIGHT +
        content_score * Config.CONTENT_SIMILARITY_WEIGHT +
        keyword_score * Config.KEYWORD_DENSITY_WEIGHT
    )

    strengths = []
    if len(matched) >= 5:
        strengths.append(f"Strong skill match with {len(matched)} matching skills")
    elif len(matched) >= 1:
        strengths.append(f"Found {len(matched)} matching skills")
    if content_score >= 50:
        strengths.append("Good content alignment with job description")
    if len(resume_skills) >= 10:
        strengths.append(f"Diverse skill set with {len(resume_skills)} skills")
    if not strengths:
        strengths.append("Resume contains professional content")

    weaknesses = []
    if len(missing) > 5:
        weaknesses.append(f"Missing {len(missing)} required skills")
    elif len(missing) > 0:
        weaknesses.append(f"Missing {len(missing)} skills from job description")
    if overall < 50:
        weaknesses.append("Overall alignment needs improvement")
    if not weaknesses:
        weaknesses.append("No major weaknesses identified")

    recommendations = []
    if missing:
        top_missing = list(missing)[:5]
        recommendations.append(f"Add these skills: {', '.join(top_missing)}")
    if overall < 60:
        recommendations.append("Use more keywords from the job description")
    recommendations.append("Quantify achievements with numbers")
    recommendations.append("Tailor resume for each application")

    return AnalysisResult(
        overall_score=round(overall, 1),
        skill_match_score=round(skill_score, 1),
        content_similarity_score=round(content_score, 1),
        keyword_match_score=round(keyword_score, 1),
        matched_skills=matched,
        missing_skills=missing,
        resume_only_skills=extra,
        matched_keywords=matched_kw[:10],
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations[:5],
        skill_categories=categorize_skills(matched),
        missing_categories=categorize_skills(missing)
    )

# ============================================================================
# AI RESUME GENERATOR (Claude API)
# ============================================================================

def generate_improved_resume(
    resume_text: str,
    job_description: str,
    result: AnalysisResult,
    api_key: str
) -> str:
    """Use Claude to generate a tailored, improved resume."""
    client = anthropic.Anthropic(api_key=api_key)

    missing_skills_str = ", ".join(sorted(result.missing_skills)) if result.missing_skills else "None"
    matched_skills_str = ", ".join(sorted(result.matched_skills)) if result.matched_skills else "None"

    prompt = f"""You are an expert resume writer and ATS optimization specialist.

I have a candidate's resume and a job description. Your task is to rewrite the resume so it:
1. Naturally incorporates the missing skills where relevant (only if they can be honestly inferred or are learnable)
2. Strengthens the existing matched skills with better phrasing
3. Uses keywords from the job description throughout
4. Quantifies achievements with realistic numbers where possible
5. Follows a clean, ATS-friendly format
6. Keeps all real experience, education, and personal details EXACTLY as-is

--- ORIGINAL RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}

--- ANALYSIS CONTEXT ---
Already Matched Skills: {matched_skills_str}
Missing Skills to Incorporate: {missing_skills_str}
Current ATS Score: {result.overall_score}/100

--- INSTRUCTIONS ---
- Output ONLY the final resume text, no commentary or preamble
- Use clear section headers: CONTACT, PROFESSIONAL SUMMARY, SKILLS, EXPERIENCE, EDUCATION, CERTIFICATIONS (if any)
- In SKILLS section, list all matched + missing skills in a clean comma-separated or categorized format
- In PROFESSIONAL SUMMARY, mention the role from the job description and key matching skills
- In EXPERIENCE, weave in JD keywords naturally into bullet points
- Use strong action verbs: Led, Developed, Implemented, Optimized, Architected, Delivered, etc.
- Each experience bullet should ideally have: Action + Task + Result/Impact
- Keep the tone professional and concise
- Do NOT fabricate companies, degrees, or dates — only enhance descriptions

Output the complete improved resume now:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ============================================================================
# PDF GENERATION — ANALYSIS REPORT
# ============================================================================

def generate_pdf_report(result: AnalysisResult, resume_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
    )

    title_style = ParagraphStyle('CustomTitle', fontName='Helvetica-Bold', fontSize=22,
        leading=26, alignment=TA_CENTER, spaceAfter=20, textColor=colors.HexColor('#1a1a2e'))
    heading_style = ParagraphStyle('CustomHeading', fontName='Helvetica-Bold', fontSize=14,
        leading=18, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#16213e'))
    body_style = ParagraphStyle('CustomBody', fontName='Helvetica', fontSize=10,
        leading=14, spaceAfter=5, alignment=TA_LEFT)
    center_style = ParagraphStyle('CustomCenter', fontName='Helvetica-Bold', fontSize=28,
        leading=32, alignment=TA_CENTER, spaceAfter=10)
    small_center = ParagraphStyle('CustomSmallCenter', fontName='Helvetica', fontSize=12,
        alignment=TA_CENTER, spaceAfter=15, textColor=colors.gray)
    footer_style = ParagraphStyle('Footer', fontName='Helvetica', fontSize=8,
        alignment=TA_CENTER, textColor=colors.gray)

    story = []
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 15))

    date_str = datetime.datetime.now().strftime('%B %d, %Y at %H:%M')
    story.append(Paragraph(f"<b>Analysis Date:</b> {date_str}", body_style))
    story.append(Paragraph(f"<b>Resume File:</b> {resume_name}", body_style))
    story.append(Paragraph(f"<b>Generated by:</b> AI Resume Analyzer v{Config.VERSION}", body_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Overall ATS Score", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 10))

    score_color = '#28a745' if result.overall_score >= 70 else '#ffc107' if result.overall_score >= 50 else '#dc3545'
    score_label = 'Excellent' if result.overall_score >= 80 else 'Good' if result.overall_score >= 60 else 'Fair' if result.overall_score >= 40 else 'Needs Work'

    story.append(Paragraph(f"<font color='{score_color}'>{result.overall_score}/100</font>", center_style))
    story.append(Paragraph(f"<font color='{score_color}'>{score_label} Match</font>", small_center))
    story.append(Spacer(1, 15))

    score_data = [
        ['Component', 'Score', 'Weight'],
        ['Skill Match', f"{result.skill_match_score}%", '40%'],
        ['Content Similarity', f"{result.content_similarity_score}%", '35%'],
        ['Keyword Match', f"{result.keyword_match_score}%", '25%'],
    ]
    table = Table(score_data, colWidths=[180, 100, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 25))

    for section_title, items, prefix in [
        ("Matched Skills", sorted(result.matched_skills), None),
        ("Missing Skills", sorted(result.missing_skills), None),
        ("Strengths", result.strengths, "✓"),
        ("Areas for Improvement", result.weaknesses, "!"),
        ("Recommendations", result.recommendations, None),
    ]:
        story.append(Paragraph(section_title, heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 8))
        if items:
            if section_title in ("Matched Skills", "Missing Skills"):
                count = len(result.matched_skills) if section_title == "Matched Skills" else len(result.missing_skills)
                label = "skills matched" if section_title == "Matched Skills" else "skills to add"
                story.append(Paragraph(f"<b>{count}</b> {label}:", body_style))
                story.append(Paragraph(", ".join(items), body_style))
            elif section_title == "Recommendations":
                for i, item in enumerate(items, 1):
                    story.append(Paragraph(f"{i}. {item}", body_style))
            else:
                for item in items:
                    p = f"{prefix} {item}" if prefix else item
                    story.append(Paragraph(p, body_style))
        else:
            msg = "All skills matched!" if section_title == "Missing Skills" else "No matching skills found."
            story.append(Paragraph(msg, body_style))
        story.append(Spacer(1, 15))

    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Generated by AI Resume Analyzer — Optimize your resume for success!", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# PDF GENERATION — IMPROVED RESUME
# ============================================================================

def generate_resume_pdf(resume_text: str, candidate_name: str = "Candidate") -> bytes:
    """Convert the AI-generated resume text into a clean, styled PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=55, leftMargin=55, topMargin=50, bottomMargin=50
    )

    # Styles
    name_style = ParagraphStyle('Name', fontName='Helvetica-Bold', fontSize=20,
        leading=24, alignment=TA_CENTER, spaceAfter=4, textColor=colors.HexColor('#1a1a2e'))
    contact_style = ParagraphStyle('Contact', fontName='Helvetica', fontSize=9,
        leading=13, alignment=TA_CENTER, spaceAfter=12, textColor=colors.HexColor('#555555'))
    section_style = ParagraphStyle('Section', fontName='Helvetica-Bold', fontSize=11,
        leading=14, spaceBefore=14, spaceAfter=4, textColor=colors.HexColor('#1a1a2e'))
    body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=9.5,
        leading=14, spaceAfter=4, alignment=TA_LEFT)
    bullet_style = ParagraphStyle('Bullet', fontName='Helvetica', fontSize=9.5,
        leading=13, spaceAfter=3, leftIndent=14, bulletIndent=4)
    job_title_style = ParagraphStyle('JobTitle', fontName='Helvetica-Bold', fontSize=10,
        leading=13, spaceAfter=1, textColor=colors.HexColor('#16213e'))
    date_style = ParagraphStyle('Date', fontName='Helvetica-Oblique', fontSize=9,
        leading=12, spaceAfter=4, textColor=colors.HexColor('#777777'))
    footer_style = ParagraphStyle('Footer', fontName='Helvetica', fontSize=8,
        alignment=TA_CENTER, textColor=colors.gray)

    story = []
    lines = resume_text.strip().split('\n')

    # Parse sections from the generated resume text
    sections = {}
    current_section = "HEADER"
    current_lines = []

    SECTION_KEYWORDS = [
        "CONTACT", "PROFESSIONAL SUMMARY", "SUMMARY", "OBJECTIVE",
        "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS",
        "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
        "EDUCATION", "CERTIFICATIONS", "PROJECTS", "ACHIEVEMENTS", "AWARDS"
    ]

    def is_section_header(line: str) -> bool:
        stripped = line.strip().upper()
        for kw in SECTION_KEYWORDS:
            if stripped == kw or stripped.startswith(kw):
                return True
        return False

    for line in lines:
        if is_section_header(line):
            if current_lines:
                sections[current_section] = current_lines
            current_section = line.strip().upper()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections[current_section] = current_lines

    # --- HEADER: Extract name & contact from first lines ---
    header_lines = sections.get("HEADER", [])
    name_line = ""
    contact_lines = []
    for i, ln in enumerate(header_lines):
        stripped = ln.strip()
        if not stripped:
            continue
        if i == 0 and not any(c in stripped for c in ['@', '|', '+', 'Phone', 'Email', 'linkedin']):
            name_line = stripped
        else:
            contact_lines.append(stripped)

    if name_line:
        story.append(Paragraph(name_line, name_style))
    if contact_lines:
        story.append(Paragraph(" &nbsp;|&nbsp; ".join(contact_lines), contact_style))

    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 8))

    # --- Render remaining sections ---
    section_order = [
        "PROFESSIONAL SUMMARY", "SUMMARY", "OBJECTIVE",
        "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS",
        "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
        "EDUCATION",
        "CERTIFICATIONS", "PROJECTS", "ACHIEVEMENTS", "AWARDS"
    ]

    # Add any sections not in the predefined order at the end
    all_section_keys = list(sections.keys())
    remaining = [k for k in all_section_keys if k != "HEADER" and k not in section_order]
    ordered_keys = [k for k in section_order if k in sections] + remaining

    for sec_key in ordered_keys:
        sec_lines = sections[sec_key]
        # Section title
        display_title = sec_key.title().replace("And", "and")
        story.append(Paragraph(display_title, section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 4))

        for ln in sec_lines:
            stripped = ln.strip()
            if not stripped:
                story.append(Spacer(1, 3))
                continue

            # Bullet points
            if stripped.startswith(('•', '-', '*', '–')):
                text = stripped.lstrip('•-*– ').strip()
                story.append(Paragraph(f"• {text}", bullet_style))
            # Bold job titles / company lines (heuristic: all caps or has year)
            elif re.search(r'\b(19|20)\d{2}\b', stripped) and len(stripped) < 100:
                story.append(Paragraph(stripped, date_style))
            elif stripped.isupper() and len(stripped) < 80:
                story.append(Paragraph(stripped, job_title_style))
            # Skills lines (comma separated)
            elif ',' in stripped and len(stripped.split(',')) > 3:
                story.append(Paragraph(stripped, body_style))
            else:
                story.append(Paragraph(stripped, body_style))

        story.append(Spacer(1, 6))

    # Footer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 6))
    date_str = datetime.datetime.now().strftime('%B %d, %Y')
    story.append(Paragraph(f"Resume generated by AI Resume Analyzer on {date_str}", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# STREAMLIT UI
# ============================================================================

def get_score_color(score):
    if score >= 80: return '#28a745'
    if score >= 60: return '#5cb85c'
    if score >= 40: return '#ffc107'
    return '#dc3545'

def get_score_label(score):
    if score >= 80: return 'Excellent Match'
    if score >= 60: return 'Good Match'
    if score >= 40: return 'Fair Match'
    return 'Needs Improvement'

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
    .main-header p { color: #a0a0a0; margin: 0.5rem 0 0 0; }
    .score-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;
    }
    .score-value { font-size: 4rem; font-weight: bold; margin: 0; }
    .score-label { color: #a0a0a0; font-size: 1.2rem; }
    .metric-box {
        background: #1a1a2e; padding: 1.5rem; border-radius: 10px; text-align: center;
    }
    .metric-box h3 { margin: 0 0 0.5rem 0; }
    .metric-box p { color: #a0a0a0; margin: 0; }
    .skill-tag {
        display: inline-block; padding: 0.4rem 0.8rem; margin: 0.2rem;
        border-radius: 20px; font-size: 0.85rem;
    }
    .skill-matched { background: #28a74533; color: #28a745; border: 1px solid #28a745; }
    .skill-missing { background: #dc354533; color: #dc3545; border: 1px solid #dc3545; }
    .skill-extra { background: #17a2b833; color: #17a2b8; border: 1px solid #17a2b8; }
    .insight-box {
        background: #16213e; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .insight-good { border-left-color: #28a745; }
    .insight-warn { border-left-color: #ffc107; }
    .insight-info { border-left-color: #17a2b8; }
    .resume-gen-box {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border: 1px solid #e94560; padding: 2rem; border-radius: 15px; margin: 1.5rem 0;
    }
    .resume-gen-box h2 { color: #e94560; margin: 0 0 0.5rem 0; }
    .resume-gen-box p { color: #a0a0a0; margin: 0; }
    .api-key-box {
        background: #16213e; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
        border: 1px solid #0f3460;
    }
    .resume-preview {
        background: #f8f9fa; color: #1a1a2e; padding: 2rem; border-radius: 10px;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
        line-height: 1.6; white-space: pre-wrap; max-height: 600px; overflow-y: auto;
        border: 1px solid #dee2e6;
    }
    .gen-badge {
        display: inline-block; background: #e94560; color: white;
        padding: 0.2rem 0.7rem; border-radius: 12px; font-size: 0.75rem;
        font-weight: bold; margin-left: 8px; vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 AI Resume Analyzer</h1>
    <p>Professional ATS Analysis + AI-Powered Resume Generation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 How to Use")
    st.markdown("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze**
    4. Review your ATS results
    5. Enter API key & generate your improved resume
    6. Download both reports!
    """)
    st.markdown("---")
    st.markdown("## 📊 Scoring Weights")
    st.markdown("""
    - **Skill Match** (40%): Skills coverage
    - **Content** (35%): Text similarity  
    - **Keywords** (25%): Keyword match
    """)
    st.markdown("---")
    st.markdown("## 🎯 Score Guide")
    st.markdown("""
    - 80–100: Excellent ✅
    - 60–79: Good 👍
    - 40–59: Fair ⚠️
    - 0–39: Needs Work ❌
    """)
    st.markdown("---")
    st.markdown("## ✨ What's New")
    st.markdown("""
    **v2.0** — AI Resume Generator!  
    After analysis, Claude rewrites your resume  
    tailored to the job description 🚀
    """)

# Session state
for key in ['result', 'resume_name', 'resume_text', 'generated_resume']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── INPUT SECTION ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'])
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

with col2:
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area(
        "Paste job description", height=200,
        placeholder="Paste the full job description here..."
    )

st.markdown("---")

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True, type="primary")

# ── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume PDF")
    elif not job_desc or len(job_desc.strip()) < 50:
        st.error("⚠️ Please enter a job description (at least 50 characters)")
    else:
        try:
            with st.spinner("Analyzing your resume..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                if len(resume_text.strip()) < 50:
                    st.error("⚠️ Could not extract enough text from PDF")
                else:
                    result = analyze_resume(resume_text, job_desc)
                    st.session_state.result = result
                    st.session_state.resume_name = uploaded_file.name
                    st.session_state.resume_text = resume_text
                    st.session_state.generated_resume = None  # reset on new analysis
                    st.success("✅ Analysis complete! Scroll down to see results.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ── DISPLAY ANALYSIS RESULTS ─────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    st.markdown("---")
    st.markdown("## 📊 ATS Analysis Results")

    color = get_score_color(result.overall_score)
    label = get_score_label(result.overall_score)
    st.markdown(f"""
    <div class="score-box">
        <p class="score-value" style="color: {color};">{result.overall_score}</p>
        <p class="score-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(result.overall_score / 100)

    m1, m2, m3 = st.columns(3)
    for col, score, label_text in [
        (m1, result.skill_match_score, "Skill Match"),
        (m2, result.content_similarity_score, "Content Similarity"),
        (m3, result.keyword_match_score, "Keyword Match"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: {get_score_color(score)};">{score}%</h3>
                <p>{label_text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Skills
    st.markdown("### 📊 Skills Analysis")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### ✅ Matched Skills")
        if result.matched_skills:
            html = ''.join([f'<span class="skill-tag skill-matched">{s}</span>' for s in sorted(result.matched_skills)])
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No matching skills found")
    with s2:
        st.markdown("#### ❌ Missing Skills")
        if result.missing_skills:
            html = ''.join([f'<span class="skill-tag skill-missing">{s}</span>' for s in sorted(result.missing_skills)])
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.success("All skills matched!")

    if result.resume_only_skills:
        st.markdown("#### 🌟 Additional Skills (Not in JD)")
        html = ''.join([f'<span class="skill-tag skill-extra">{s}</span>' for s in sorted(list(result.resume_only_skills)[:15])])
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # Insights
    st.markdown("### 💡 Insights")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("#### 💪 Strengths")
        for s in result.strengths:
            st.markdown(f'<div class="insight-box insight-good">✓ {s}</div>', unsafe_allow_html=True)
    with i2:
        st.markdown("#### ⚠️ Areas to Improve")
        for w in result.weaknesses:
            st.markdown(f'<div class="insight-box insight-warn">! {w}</div>', unsafe_allow_html=True)

    st.markdown("#### 🎯 Recommendations")
    for i, r in enumerate(result.recommendations, 1):
        st.markdown(f'<div class="insight-box insight-info">{i}. {r}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Download analysis report
    st.markdown("### 📥 Download Analysis Report")
    try:
        pdf_data = generate_pdf_report(result, st.session_state.resume_name)
        st.download_button(
            label="📥 Download Analysis Report (PDF)",
            data=pdf_data,
            file_name=f"ats_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Could not generate report PDF: {str(e)}")

    with st.expander("📈 Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matched Skills", len(result.matched_skills))
        c2.metric("Missing Skills", len(result.missing_skills))
        c3.metric("Extra Skills", len(result.resume_only_skills))
        c4.metric("Keywords Hit", len(result.matched_keywords))

    # ══════════════════════════════════════════════════════════════════════════
    # AI RESUME GENERATOR SECTION
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("""
    <div class="resume-gen-box">
        <h2>✨ AI Resume Generator <span class="gen-badge">NEW</span></h2>
        <p>Claude rewrites your resume tailored to this job — incorporating missing skills,
        job-specific keywords, and stronger phrasing. Your experience & education stay intact.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="api-key-box">', unsafe_allow_html=True)
    st.markdown("#### 🔑 Anthropic API Key")
    st.markdown("Enter your Anthropic API key to generate an improved resume. "
                "[Get one here →](https://console.anthropic.com/)")
    api_key = st.text_input(
        "API Key", type="password",
        placeholder="sk-ant-api...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col_gen, col_note = st.columns([1, 2])
    with col_gen:
        generate_btn = st.button(
            "🚀 Generate Improved Resume",
            use_container_width=True,
            type="primary",
            disabled=not api_key
        )
    with col_note:
        st.caption("⚡ Uses Claude claude-sonnet-4-20250514 · Takes ~15–30 seconds · Your key is never stored")

    if generate_btn:
        if not api_key or not api_key.startswith("sk-"):
            st.error("⚠️ Please enter a valid Anthropic API key (starts with 'sk-')")
        else:
            with st.spinner("✨ Claude is rewriting your resume... This may take 20–30 seconds."):
                try:
                    improved = generate_improved_resume(
                        st.session_state.resume_text,
                        job_desc,
                        result,
                        api_key
                    )
                    st.session_state.generated_resume = improved
                    st.success("🎉 Resume generated successfully!")
                except anthropic.AuthenticationError:
                    st.error("❌ Invalid API key. Please check your Anthropic API key and try again.")
                except anthropic.RateLimitError:
                    st.error("❌ Rate limit reached. Please wait a moment and try again.")
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")

    # ── SHOW GENERATED RESUME ─────────────────────────────────────────────────
    if st.session_state.generated_resume:
        st.markdown("---")
        st.markdown("### 📄 Your Improved Resume")

        # Quick stats
        orig_skills = len(result.matched_skills)
        all_skills_in_new = len(result.matched_skills) + len(result.missing_skills)
        q1, q2, q3 = st.columns(3)
        q1.metric("Skills Before", orig_skills, delta=None)
        q2.metric("Skills After", all_skills_in_new,
                  delta=f"+{len(result.missing_skills)} added")
        q3.metric("Missing Skills Addressed", len(result.missing_skills))

        # Preview
        with st.expander("👁️ Preview Generated Resume", expanded=True):
            st.markdown(
                f'<div class="resume-preview">{st.session_state.generated_resume}</div>',
                unsafe_allow_html=True
            )

        # Edit option
        with st.expander("✏️ Edit Before Downloading"):
            edited_resume = st.text_area(
                "Edit your resume here:",
                value=st.session_state.generated_resume,
                height=500,
                label_visibility="collapsed"
            )
            if st.button("💾 Save Edits"):
                st.session_state.generated_resume = edited_resume
                st.success("✅ Edits saved!")

        # Downloads
        st.markdown("### 📥 Download Improved Resume")
        dl1, dl2 = st.columns(2)

        with dl1:
            try:
                resume_pdf = generate_resume_pdf(
                    st.session_state.generated_resume,
                    st.session_state.resume_name.replace('.pdf', '')
                )
                st.download_button(
                    label="📄 Download Resume as PDF",
                    data=resume_pdf,
                    file_name=f"improved_resume_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")

        with dl2:
            st.download_button(
                label="📝 Download Resume as TXT",
                data=st.session_state.generated_resume.encode('utf-8'),
                file_name=f"improved_resume_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.info("💡 **Tip:** The TXT version is easy to paste into LinkedIn, job portals, or Word. "
                "Review the resume and verify all details before submitting.")
