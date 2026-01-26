# 📄 AI Resume Analyzer

A professional, production-grade ATS (Applicant Tracking System) Resume Analyzer built with Python and Streamlit. This tool helps job seekers optimize their resumes by analyzing them against job descriptions using advanced NLP techniques.

## 🚀 Features

- **PDF Resume Parsing**: Extract text from PDF resumes
- **Skill Extraction**: Identify 500+ skills across 13 categories
- **ATS Scoring**: TF-IDF vectorization with cosine similarity
- **Comprehensive Analysis**: Matched/missing skills, strengths, weaknesses
- **PDF Report Generation**: Professional downloadable reports
- **Dark Theme UI**: Modern, professional interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **NLP**: NLTK, scikit-learn
- **PDF Processing**: PyPDF2
- **Report Generation**: ReportLab
- **Deployment**: Streamlit Cloud

## 📦 Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
