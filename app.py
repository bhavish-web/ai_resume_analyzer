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
    VERSION = "1.0.0"
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
# PDF REPORT - FIXED VERSION
# ============================================================================

def generate_pdf_report(result: AnalysisResult, resume_name: str) -> bytes:
    """Generate PDF report with fresh styles each time."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4, 
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50
    )
    
    # Create fresh styles each time (not using getSampleStyleSheet)
    title_style = ParagraphStyle(
        'CustomTitle',
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor('#1a1a2e')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#16213e')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceAfter=5,
        alignment=TA_LEFT
    )
    
    center_style = ParagraphStyle(
        'CustomCenter',
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=32,
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    small_center = ParagraphStyle(
        'CustomSmallCenter',
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=15,
        textColor=colors.gray
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 15))
    
    # Meta info
    date_str = datetime.datetime.now().strftime('%B %d, %Y at %H:%M')
    story.append(Paragraph(f"<b>Analysis Date:</b> {date_str}", body_style))
    story.append(Paragraph(f"<b>Resume File:</b> {resume_name}", body_style))
    story.append(Paragraph(f"<b>Generated by:</b> AI Resume Analyzer v{Config.VERSION}", body_style))
    story.append(Spacer(1, 20))
    
    # Score section
    story.append(Paragraph("Overall ATS Score", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 10))
    
    score_color = '#28a745' if result.overall_score >= 70 else '#ffc107' if result.overall_score >= 50 else '#dc3545'
    score_label = 'Excellent' if result.overall_score >= 80 else 'Good' if result.overall_score >= 60 else 'Fair' if result.overall_score >= 40 else 'Needs Work'
    
    story.append(Paragraph(f"<font color='{score_color}'>{result.overall_score}/100</font>", center_style))
    story.append(Paragraph(f"<font color='{score_color}'>{score_label} Match</font>", small_center))
    story.append(Spacer(1, 15))
    
    # Score breakdown table
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
    
    # Matched Skills
    story.append(Paragraph("Matched Skills", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 8))
    
    if result.matched_skills:
        story.append(Paragraph(f"<b>{len(result.matched_skills)}</b> skills matched:", body_style))
        skills_text = ", ".join(sorted(result.matched_skills))
        story.append(Paragraph(skills_text, body_style))
    else:
        story.append(Paragraph("No matching skills found.", body_style))
    story.append(Spacer(1, 15))
    
    # Missing Skills
    story.append(Paragraph("Missing Skills", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 8))
    
    if result.missing_skills:
        story.append(Paragraph(f"<b>{len(result.missing_skills)}</b> skills to add:", body_style))
        skills_text = ", ".join(sorted(result.missing_skills))
        story.append(Paragraph(skills_text, body_style))
    else:
        story.append(Paragraph("All required skills are present!", body_style))
    story.append(Spacer(1, 15))
    
    # Strengths
    story.append(Paragraph("Strengths", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 8))
    
    for strength in result.strengths:
        story.append(Paragraph(f"+ {strength}", body_style))
    story.append(Spacer(1, 15))
    
    # Areas for Improvement
    story.append(Paragraph("Areas for Improvement", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 8))
    
    for weakness in result.weaknesses:
        story.append(Paragraph(f"- {weakness}", body_style))
    story.append(Spacer(1, 15))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 8))
    
    for i, rec in enumerate(result.recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
    story.append(Spacer(1, 25))
    
    # Footer
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 10))
    
    footer_style = ParagraphStyle(
        'Footer',
        fontName='Helvetica',
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.gray
    )
    story.append(Paragraph("Generated by AI Resume Analyzer - Optimize your resume for success!", footer_style))
    
    # Build PDF
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 AI Resume Analyzer</h1>
    <p>Professional ATS-Style Resume Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 How to Use")
    st.markdown("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click Analyze
    4. Review your results
    5. Download the PDF report
    """)
    st.markdown("---")
    st.markdown("## 📊 Scoring")
    st.markdown("""
    - **Skill Match** (40%): Skills coverage
    - **Content** (35%): Text similarity  
    - **Keywords** (25%): Keyword match
    """)
    st.markdown("---")
    st.markdown("## 🎯 Score Guide")
    st.markdown("""
    - 80-100: Excellent ✅
    - 60-79: Good 👍
    - 40-59: Fair ⚠️
    - 0-39: Needs Work ❌
    """)

# Session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'resume_name' not in st.session_state:
    st.session_state.resume_name = None

# Input section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'])
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

with col2:
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area("Paste job description", height=200, placeholder="Paste the job description here...")

st.markdown("---")

# Analyze button
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True, type="primary")

# Run analysis
if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume PDF")
    elif not job_desc or len(job_desc.strip()) < 50:
        st.error("⚠️ Please enter a job description (at least 50 characters)")
    else:
        try:
            with st.spinner("Analyzing..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                if len(resume_text.strip()) < 50:
                    st.error("⚠️ Could not extract enough text from PDF")
                else:
                    result = analyze_resume(resume_text, job_desc)
                    st.session_state.result = result
                    st.session_state.resume_name = uploaded_file.name
                    st.success("✅ Analysis complete!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.result:
    result = st.session_state.result
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Score
    color = get_score_color(result.overall_score)
    label = get_score_label(result.overall_score)
    st.markdown(f"""
    <div class="score-box">
        <p class="score-value" style="color: {color};">{result.overall_score}</p>
        <p class="score-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(result.overall_score / 100)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {get_score_color(result.skill_match_score)};">{result.skill_match_score}%</h3>
            <p>Skill Match</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {get_score_color(result.content_similarity_score)};">{result.content_similarity_score}%</h3>
            <p>Content Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {get_score_color(result.keyword_match_score)};">{result.keyword_match_score}%</h3>
            <p>Keyword Match</p>
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
        st.markdown("#### 🌟 Additional Skills")
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
        st.markdown("#### ⚠️ Improve")
        for w in result.weaknesses:
            st.markdown(f'<div class="insight-box insight-warn">! {w}</div>', unsafe_allow_html=True)
    
    st.markdown("#### 🎯 Recommendations")
    for i, r in enumerate(result.recommendations, 1):
        st.markdown(f'<div class="insight-box insight-info">{i}. {r}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download
    st.markdown("### 📥 Download Report")
    try:
        pdf_data = generate_pdf_report(result, st.session_state.resume_name)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name=f"resume_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Could not generate PDF: {str(e)}")
    
    # Stats
    with st.expander("📈 Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matched", len(result.matched_skills))
        c2.metric("Missing", len(result.missing_skills))
        c3.metric("Extra", len(result.resume_only_skills))
        c4.metric("Keywords", len(result.matched_keywords))
