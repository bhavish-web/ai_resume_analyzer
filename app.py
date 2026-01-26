"""
================================================================================
AI RESUME ANALYZER - PRODUCTION QUALITY ATS SYSTEM
================================================================================
A flagship portfolio project demonstrating advanced NLP, document processing,
and machine learning techniques for resume-job description matching.

Author: AI Resume Analyzer Team
Version: 1.0.0
License: MIT
================================================================================
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import streamlit as st
import io
import re
import string
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import warnings

# PDF Processing
import pdfplumber
from PyPDF2 import PdfReader

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF Report Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# NLTK DATA DOWNLOAD (Required for first run)
# =============================================================================

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data packages."""
    packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    return True

# Initialize NLTK data
download_nltk_data()

# =============================================================================
# DATA: COMPREHENSIVE SKILLS DATABASE
# =============================================================================

class SkillsDatabase:
    """
    Comprehensive skills database containing technical skills, soft skills,
    tools, frameworks, and industry-specific terminology.
    """
    
    # Programming Languages
    PROGRAMMING_LANGUAGES: Set[str] = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c',
        'ruby', 'go', 'golang', 'rust', 'swift', 'kotlin', 'scala',
        'php', 'perl', 'r', 'matlab', 'julia', 'haskell', 'erlang',
        'elixir', 'clojure', 'f#', 'objective-c', 'dart', 'lua',
        'groovy', 'cobol', 'fortran', 'assembly', 'vba', 'sql',
        'plsql', 'tsql', 'bash', 'powershell', 'shell', 'solidity'
    }
    
    # Web Technologies
    WEB_TECHNOLOGIES: Set[str] = {
        'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less',
        'bootstrap', 'tailwind', 'tailwindcss', 'materialize',
        'jquery', 'ajax', 'json', 'xml', 'rest', 'restful',
        'graphql', 'websocket', 'webrtc', 'pwa', 'spa', 'ssr',
        'responsive design', 'web accessibility', 'wcag', 'seo'
    }
    
    # Frontend Frameworks
    FRONTEND_FRAMEWORKS: Set[str] = {
        'react', 'reactjs', 'react.js', 'angular', 'angularjs',
        'vue', 'vuejs', 'vue.js', 'svelte', 'ember', 'emberjs',
        'backbone', 'backbonejs', 'next.js', 'nextjs', 'nuxt',
        'nuxtjs', 'gatsby', 'redux', 'mobx', 'vuex', 'ngrx',
        'material-ui', 'mui', 'ant design', 'chakra ui', 'styled-components'
    }
    
    # Backend Frameworks
    BACKEND_FRAMEWORKS: Set[str] = {
        'django', 'flask', 'fastapi', 'express', 'expressjs',
        'node', 'nodejs', 'node.js', 'spring', 'spring boot',
        'springboot', 'rails', 'ruby on rails', 'laravel',
        'symfony', 'asp.net', '.net', '.net core', 'dotnet',
        'gin', 'echo', 'fiber', 'actix', 'rocket', 'phoenix',
        'koa', 'nestjs', 'hapi', 'tornado', 'pyramid', 'bottle',
        'aiohttp', 'sanic', 'starlette', 'quart'
    }
    
    # Databases
    DATABASES: Set[str] = {
        'mysql', 'postgresql', 'postgres', 'mongodb', 'redis',
        'sqlite', 'oracle', 'sql server', 'mssql', 'mariadb',
        'cassandra', 'dynamodb', 'couchdb', 'couchbase', 'neo4j',
        'elasticsearch', 'opensearch', 'influxdb', 'timescaledb',
        'cockroachdb', 'firestore', 'firebase', 'supabase',
        'prisma', 'sequelize', 'typeorm', 'mongoose', 'sqlalchemy',
        'hibernate', 'jpa', 'entity framework', 'dapper'
    }
    
    # Cloud Platforms & Services
    CLOUD_PLATFORMS: Set[str] = {
        'aws', 'amazon web services', 'azure', 'microsoft azure',
        'gcp', 'google cloud', 'google cloud platform', 'heroku',
        'digitalocean', 'linode', 'vultr', 'cloudflare', 'vercel',
        'netlify', 'railway', 'render', 'fly.io', 'oracle cloud',
        'ibm cloud', 'alibaba cloud', 'openstack'
    }
    
    # AWS Services
    AWS_SERVICES: Set[str] = {
        'ec2', 'ecs', 'eks', 'lambda', 's3', 'rds', 'dynamodb',
        'cloudfront', 'route53', 'vpc', 'iam', 'cognito', 'sns',
        'sqs', 'kinesis', 'redshift', 'athena', 'glue', 'emr',
        'sagemaker', 'cloudwatch', 'cloudformation', 'cdk',
        'api gateway', 'step functions', 'eventbridge', 'secrets manager'
    }
    
    # DevOps & Infrastructure
    DEVOPS_TOOLS: Set[str] = {
        'docker', 'kubernetes', 'k8s', 'helm', 'terraform',
        'ansible', 'puppet', 'chef', 'vagrant', 'packer',
        'jenkins', 'gitlab ci', 'github actions', 'circleci',
        'travis ci', 'bamboo', 'teamcity', 'argocd', 'spinnaker',
        'prometheus', 'grafana', 'datadog', 'splunk', 'elk',
        'elasticsearch', 'logstash', 'kibana', 'fluentd', 'loki',
        'jaeger', 'zipkin', 'opentelemetry', 'nagios', 'zabbix',
        'nginx', 'apache', 'haproxy', 'traefik', 'envoy', 'istio',
        'consul', 'vault', 'etcd', 'rancher', 'openshift'
    }
    
    # Data Science & Machine Learning
    DATA_SCIENCE: Set[str] = {
        'machine learning', 'ml', 'deep learning', 'dl', 'neural networks',
        'artificial intelligence', 'ai', 'data science', 'data analysis',
        'data analytics', 'data mining', 'data visualization',
        'statistical analysis', 'predictive modeling', 'nlp',
        'natural language processing', 'computer vision', 'cv',
        'reinforcement learning', 'supervised learning', 'unsupervised learning',
        'feature engineering', 'model optimization', 'hyperparameter tuning',
        'a/b testing', 'experimentation', 'time series analysis',
        'recommendation systems', 'anomaly detection', 'clustering',
        'classification', 'regression', 'ensemble methods'
    }
    
    # Data Science Libraries & Frameworks
    DATA_SCIENCE_TOOLS: Set[str] = {
        'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn',
        'tensorflow', 'keras', 'pytorch', 'torch', 'jax',
        'xgboost', 'lightgbm', 'catboost', 'statsmodels', 'spacy',
        'nltk', 'gensim', 'huggingface', 'transformers', 'bert',
        'gpt', 'opencv', 'pillow', 'matplotlib', 'seaborn',
        'plotly', 'bokeh', 'altair', 'd3', 'd3.js', 'tableau',
        'power bi', 'looker', 'metabase', 'superset', 'dbt',
        'airflow', 'luigi', 'prefect', 'dagster', 'mlflow',
        'kubeflow', 'weights & biases', 'wandb', 'optuna',
        'ray', 'dask', 'spark', 'pyspark', 'hadoop', 'hive',
        'flink', 'kafka', 'beam', 'streamlit', 'gradio', 'dash'
    }
    
    # Mobile Development
    MOBILE_DEVELOPMENT: Set[str] = {
        'ios', 'android', 'react native', 'flutter', 'xamarin',
        'ionic', 'cordova', 'phonegap', 'swift', 'swiftui',
        'uikit', 'kotlin', 'jetpack compose', 'room', 'retrofit',
        'realm', 'core data', 'firebase', 'push notifications',
        'mobile app development', 'cross-platform', 'hybrid apps',
        'app store optimization', 'aso', 'mobile ui', 'mobile ux'
    }
    
    # Testing & QA
    TESTING: Set[str] = {
        'unit testing', 'integration testing', 'e2e testing',
        'end-to-end testing', 'automated testing', 'manual testing',
        'test automation', 'tdd', 'bdd', 'qa', 'quality assurance',
        'selenium', 'cypress', 'playwright', 'puppeteer', 'jest',
        'mocha', 'chai', 'jasmine', 'karma', 'pytest', 'unittest',
        'junit', 'testng', 'robot framework', 'cucumber', 'postman',
        'insomnia', 'soapui', 'jmeter', 'locust', 'gatling',
        'load testing', 'performance testing', 'security testing',
        'penetration testing', 'code coverage', 'sonarqube'
    }
    
    # Version Control & Collaboration
    VERSION_CONTROL: Set[str] = {
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'subversion',
        'mercurial', 'perforce', 'azure devops', 'tfs',
        'version control', 'source control', 'branching strategies',
        'gitflow', 'trunk-based development', 'code review',
        'pull requests', 'merge requests', 'ci/cd', 'continuous integration',
        'continuous deployment', 'continuous delivery'
    }
    
    # Agile & Project Management
    AGILE_PM: Set[str] = {
        'agile', 'scrum', 'kanban', 'lean', 'waterfall',
        'sprint planning', 'retrospective', 'standup', 'daily scrum',
        'product backlog', 'user stories', 'epics', 'story points',
        'velocity', 'burndown chart', 'jira', 'confluence', 'trello',
        'asana', 'monday.com', 'notion', 'linear', 'clickup',
        'project management', 'program management', 'pmo',
        'stakeholder management', 'risk management', 'sdlc',
        'requirements gathering', 'technical documentation'
    }
    
    # Security
    SECURITY: Set[str] = {
        'cybersecurity', 'information security', 'infosec',
        'network security', 'application security', 'appsec',
        'owasp', 'security audit', 'vulnerability assessment',
        'penetration testing', 'ethical hacking', 'soc',
        'siem', 'ids', 'ips', 'firewall', 'waf', 'vpn',
        'ssl', 'tls', 'https', 'encryption', 'cryptography',
        'authentication', 'authorization', 'oauth', 'oauth2',
        'saml', 'sso', 'jwt', 'mfa', '2fa', 'rbac', 'abac',
        'zero trust', 'devsecops', 'security compliance',
        'gdpr', 'hipaa', 'pci-dss', 'soc2', 'iso27001'
    }
    
    # Architecture & Design Patterns
    ARCHITECTURE: Set[str] = {
        'microservices', 'monolithic', 'serverless', 'event-driven',
        'domain-driven design', 'ddd', 'clean architecture',
        'hexagonal architecture', 'cqrs', 'event sourcing',
        'api design', 'system design', 'software architecture',
        'enterprise architecture', 'solution architecture',
        'design patterns', 'solid principles', 'dry', 'kiss',
        'yagni', 'mvc', 'mvvm', 'mvp', 'repository pattern',
        'factory pattern', 'singleton', 'observer pattern',
        'dependency injection', 'inversion of control', 'ioc'
    }
    
    # Soft Skills
    SOFT_SKILLS: Set[str] = {
        'leadership', 'team leadership', 'people management',
        'communication', 'verbal communication', 'written communication',
        'presentation skills', 'public speaking', 'storytelling',
        'problem solving', 'problem-solving', 'critical thinking',
        'analytical thinking', 'analytical skills', 'decision making',
        'strategic thinking', 'strategic planning', 'innovation',
        'creativity', 'adaptability', 'flexibility', 'resilience',
        'time management', 'prioritization', 'organization',
        'attention to detail', 'multitasking', 'self-motivated',
        'self-starter', 'proactive', 'initiative', 'ownership',
        'accountability', 'teamwork', 'collaboration', 'team player',
        'cross-functional', 'interpersonal skills', 'relationship building',
        'stakeholder management', 'client management', 'customer focus',
        'customer service', 'negotiation', 'conflict resolution',
        'emotional intelligence', 'empathy', 'mentoring', 'coaching',
        'training', 'knowledge sharing', 'continuous learning',
        'growth mindset', 'fast learner', 'quick learner'
    }
    
    # Business & Domain Skills
    BUSINESS_SKILLS: Set[str] = {
        'business analysis', 'business development', 'sales',
        'marketing', 'digital marketing', 'content marketing',
        'product management', 'product development', 'product strategy',
        'market research', 'competitive analysis', 'swot analysis',
        'financial analysis', 'budgeting', 'forecasting', 'roi',
        'kpi', 'okr', 'metrics', 'data-driven', 'analytics',
        'reporting', 'dashboards', 'visualization', 'insights',
        'strategy', 'operations', 'process improvement', 'optimization',
        'automation', 'efficiency', 'scalability', 'growth',
        'revenue', 'profit', 'cost reduction', 'vendor management',
        'supply chain', 'logistics', 'ecommerce', 'fintech',
        'healthtech', 'edtech', 'saas', 'b2b', 'b2c', 'startup'
    }
    
    # Certifications (commonly mentioned)
    CERTIFICATIONS: Set[str] = {
        'aws certified', 'aws solutions architect', 'aws developer',
        'azure certified', 'azure administrator', 'azure developer',
        'gcp certified', 'google cloud certified', 'cka', 'ckad',
        'terraform certified', 'kubernetes certified', 'docker certified',
        'pmp', 'prince2', 'itil', 'six sigma', 'scrum master',
        'csm', 'psm', 'safe', 'cspo', 'cissp', 'cism', 'cisa',
        'comptia', 'security+', 'network+', 'ccna', 'ccnp',
        'rhce', 'rhcsa', 'lpic', 'oracle certified', 'java certified',
        'microsoft certified', 'mcse', 'mcsa', 'data engineer',
        'machine learning certified', 'tensorflow certified'
    }
    
    @classmethod
    def get_all_skills(cls) -> Set[str]:
        """Returns a combined set of all skills from all categories."""
        all_skills = set()
        all_skills.update(cls.PROGRAMMING_LANGUAGES)
        all_skills.update(cls.WEB_TECHNOLOGIES)
        all_skills.update(cls.FRONTEND_FRAMEWORKS)
        all_skills.update(cls.BACKEND_FRAMEWORKS)
        all_skills.update(cls.DATABASES)
        all_skills.update(cls.CLOUD_PLATFORMS)
        all_skills.update(cls.AWS_SERVICES)
        all_skills.update(cls.DEVOPS_TOOLS)
        all_skills.update(cls.DATA_SCIENCE)
        all_skills.update(cls.DATA_SCIENCE_TOOLS)
        all_skills.update(cls.MOBILE_DEVELOPMENT)
        all_skills.update(cls.TESTING)
        all_skills.update(cls.VERSION_CONTROL)
        all_skills.update(cls.AGILE_PM)
        all_skills.update(cls.SECURITY)
        all_skills.update(cls.ARCHITECTURE)
        all_skills.update(cls.SOFT_SKILLS)
        all_skills.update(cls.BUSINESS_SKILLS)
        all_skills.update(cls.CERTIFICATIONS)
        return all_skills
    
    @classmethod
    def get_skill_categories(cls) -> Dict[str, Set[str]]:
        """Returns a dictionary mapping category names to skill sets."""
        return {
            'Programming Languages': cls.PROGRAMMING_LANGUAGES,
            'Web Technologies': cls.WEB_TECHNOLOGIES,
            'Frontend Frameworks': cls.FRONTEND_FRAMEWORKS,
            'Backend Frameworks': cls.BACKEND_FRAMEWORKS,
            'Databases': cls.DATABASES,
            'Cloud Platforms': cls.CLOUD_PLATFORMS,
            'AWS Services': cls.AWS_SERVICES,
            'DevOps & Infrastructure': cls.DEVOPS_TOOLS,
            'Data Science & ML': cls.DATA_SCIENCE,
            'Data Science Tools': cls.DATA_SCIENCE_TOOLS,
            'Mobile Development': cls.MOBILE_DEVELOPMENT,
            'Testing & QA': cls.TESTING,
            'Version Control': cls.VERSION_CONTROL,
            'Agile & Project Management': cls.AGILE_PM,
            'Security': cls.SECURITY,
            'Architecture': cls.ARCHITECTURE,
            'Soft Skills': cls.SOFT_SKILLS,
            'Business Skills': cls.BUSINESS_SKILLS,
            'Certifications': cls.CERTIFICATIONS
        }


# =============================================================================
# UTILITY HELPERS
# =============================================================================

class TextUtils:
    """Utility class for text processing operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text by removing extra whitespace,
        special characters, and normalizing line breaks.
        """
        if not text:
            return ""
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers from text."""
        phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,6}[-\s\.]?[0-9]{3,6}'
        return re.sub(phone_pattern, '', text)
    
    @staticmethod
    def extract_years_of_experience(text: str) -> Optional[int]:
        """Extract years of experience from text."""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?:experience|exp)(?:\s+of)?\s*:?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|of|working)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate simple text overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


# =============================================================================
# PDF PARSER MODULE
# =============================================================================

class PDFParser:
    """
    Professional PDF parsing module with multiple extraction methods
    and robust error handling.
    """
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2
        ]
    
    def extract_text(self, pdf_file) -> Tuple[str, Dict]:
        """
        Extract text from PDF file using multiple methods.
        Returns extracted text and metadata.
        """
        text = ""
        metadata = {
            "pages": 0,
            "extraction_method": None,
            "success": False,
            "errors": []
        }
        
        for method in self.extraction_methods:
            try:
                # Reset file pointer
                pdf_file.seek(0)
                text, page_count = method(pdf_file)
                
                if text and len(text.strip()) > 50:
                    metadata["pages"] = page_count
                    metadata["extraction_method"] = method.__name__
                    metadata["success"] = True
                    break
            except Exception as e:
                metadata["errors"].append(f"{method.__name__}: {str(e)}")
                continue
        
        if not metadata["success"]:
            # Try one more time with basic extraction
            try:
                pdf_file.seek(0)
                text, page_count = self._basic_extraction(pdf_file)
                if text:
                    metadata["pages"] = page_count
                    metadata["extraction_method"] = "basic_extraction"
                    metadata["success"] = True
            except Exception as e:
                metadata["errors"].append(f"basic_extraction: {str(e)}")
        
        return text, metadata
    
    def _extract_with_pdfplumber(self, pdf_file) -> Tuple[str, int]:
        """Extract text using pdfplumber library."""
        text_content = []
        page_count = 0
        
        with pdfplumber.open(pdf_file) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        return '\n'.join(text_content), page_count
    
    def _extract_with_pypdf2(self, pdf_file) -> Tuple[str, int]:
        """Extract text using PyPDF2 library."""
        text_content = []
        
        reader = PdfReader(pdf_file)
        page_count = len(reader.pages)
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        
        return '\n'.join(text_content), page_count
    
    def _basic_extraction(self, pdf_file) -> Tuple[str, int]:
        """Fallback basic extraction method."""
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text, len(reader.pages)
        except Exception:
            return "", 0


# =============================================================================
# TEXT PREPROCESSOR MODULE
# =============================================================================

class TextPreprocessor:
    """
    Advanced NLP text preprocessing module with tokenization,
    lemmatization, and stopword removal.
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stop words to preserve
        self.preserve_words = {
            'python', 'java', 'r', 'c', 'go', 'sql', 'no', 'not',
            'aws', 'gcp', 'azure', 'ml', 'ai', 'ui', 'ux', 'qa',
            'api', 'apis', 'ci', 'cd', 'it', 'bi', 'pm'
        }
        
        # Update stop words
        self.stop_words -= self.preserve_words
    
    def preprocess(self, text: str, 
                   remove_stopwords: bool = True,
                   lemmatize: bool = True,
                   lowercase: bool = True) -> str:
        """
        Preprocess text with various NLP techniques.
        """
        if not text:
            return ""
        
        # Clean text
        text = TextUtils.clean_text(text)
        text = TextUtils.remove_urls(text)
        text = TextUtils.remove_emails(text)
        text = TextUtils.remove_phone_numbers(text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            tokens = word_tokenize(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Skip punctuation and single characters (except preserved)
            if token in string.punctuation:
                continue
            if len(token) == 1 and token.lower() not in self.preserve_words:
                continue
            
            # Skip numbers only
            if token.isdigit():
                continue
            
            # Remove stopwords
            if remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # Lemmatize
            if lemmatize:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def tokenize_for_skills(self, text: str) -> List[str]:
        """
        Tokenize text for skill extraction, preserving multi-word skills.
        """
        if not text:
            return []
        
        text = text.lower()
        
        # Extract potential multi-word phrases (bigrams and trigrams)
        words = text.split()
        tokens = []
        
        # Add unigrams
        tokens.extend(words)
        
        # Add bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            tokens.append(bigram)
        
        # Add trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            tokens.append(trigram)
        
        return tokens
    
    def get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if not text:
            return []
        
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(text)
        
        return sentences


# =============================================================================
# SKILL EXTRACTION ENGINE
# =============================================================================

@dataclass
class SkillMatch:
    """Data class representing a matched skill."""
    skill: str
    category: str
    frequency: int = 1
    context: str = ""


@dataclass
class SkillExtractionResult:
    """Data class for skill extraction results."""
    found_skills: List[SkillMatch] = field(default_factory=list)
    skill_categories: Dict[str, List[str]] = field(default_factory=dict)
    total_skills: int = 0
    
    def get_skills_list(self) -> List[str]:
        """Get flat list of skill names."""
        return [match.skill for match in self.found_skills]


class SkillEngine:
    """
    Advanced skill extraction engine using pattern matching
    and fuzzy matching techniques.
    """
    
    def __init__(self):
        self.skills_db = SkillsDatabase()
        self.all_skills = self.skills_db.get_all_skills()
        self.skill_categories = self.skills_db.get_skill_categories()
        self.preprocessor = TextPreprocessor()
        
        # Build skill variations for better matching
        self.skill_variations = self._build_skill_variations()
    
    def _build_skill_variations(self) -> Dict[str, str]:
        """Build a dictionary of skill variations to canonical forms."""
        variations = {}
        
        for skill in self.all_skills:
            # Original form
            variations[skill.lower()] = skill
            
            # Without dots
            variations[skill.lower().replace('.', '')] = skill
            
            # Without hyphens
            variations[skill.lower().replace('-', ' ')] = skill
            variations[skill.lower().replace('-', '')] = skill
            
            # Without spaces
            variations[skill.lower().replace(' ', '')] = skill
            
            # Common abbreviation handling
            if '.' in skill:
                variations[skill.lower().replace('.js', 'js')] = skill
        
        return variations
    
    def extract_skills(self, text: str) -> SkillExtractionResult:
        """
        Extract skills from text using pattern matching.
        """
        result = SkillExtractionResult()
        
        if not text:
            return result
        
        # Normalize text for matching
        text_lower = text.lower()
        
        # Track found skills to avoid duplicates
        found_skills_set = set()
        skill_counts = Counter()
        
        # Get all tokens including n-grams
        tokens = self.preprocessor.tokenize_for_skills(text_lower)
        
        # Match against skill database
        for token in tokens:
            token_clean = token.strip()
            
            # Check direct match
            if token_clean in self.skill_variations:
                canonical_skill = self.skill_variations[token_clean]
                if canonical_skill.lower() not in found_skills_set:
                    found_skills_set.add(canonical_skill.lower())
                    skill_counts[canonical_skill.lower()] += 1
            
            # Check cleaned variations
            token_no_punct = re.sub(r'[^\w\s]', '', token_clean)
            if token_no_punct in self.skill_variations:
                canonical_skill = self.skill_variations[token_no_punct]
                if canonical_skill.lower() not in found_skills_set:
                    found_skills_set.add(canonical_skill.lower())
                    skill_counts[canonical_skill.lower()] += 1
        
        # Also check for skills as substrings for multi-word skills
        for skill in self.all_skills:
            skill_lower = skill.lower()
            if len(skill_lower) > 3:  # Only for longer skills to avoid false positives
                if skill_lower in text_lower and skill_lower not in found_skills_set:
                    found_skills_set.add(skill_lower)
                    skill_counts[skill_lower] += 1
        
        # Build result
        for skill_lower in found_skills_set:
            category = self._get_skill_category(skill_lower)
            skill_match = SkillMatch(
                skill=skill_lower,
                category=category,
                frequency=skill_counts.get(skill_lower, 1)
            )
            result.found_skills.append(skill_match)
            
            # Add to category mapping
            if category not in result.skill_categories:
                result.skill_categories[category] = []
            result.skill_categories[category].append(skill_lower)
        
        result.total_skills = len(result.found_skills)
        
        return result
    
    def _get_skill_category(self, skill: str) -> str:
        """Get the category of a skill."""
        skill_lower = skill.lower()
        
        for category, skills in self.skill_categories.items():
            if skill_lower in skills:
                return category
        
        return "Other Skills"
    
    def compare_skills(self, resume_skills: List[str], 
                       jd_skills: List[str]) -> Dict[str, List[str]]:
        """
        Compare resume skills against job description skills.
        """
        resume_set = set(s.lower() for s in resume_skills)
        jd_set = set(s.lower() for s in jd_skills)
        
        matched = resume_set.intersection(jd_set)
        missing = jd_set - resume_set
        additional = resume_set - jd_set
        
        return {
            "matched": sorted(list(matched)),
            "missing": sorted(list(missing)),
            "additional": sorted(list(additional))
        }


# =============================================================================
# ATS SCORING ENGINE
# =============================================================================

@dataclass
class ATSScore:
    """Data class for ATS scoring results."""
    overall_score: float = 0.0
    skill_match_score: float = 0.0
    content_similarity_score: float = 0.0
    keyword_density_score: float = 0.0
    experience_match_score: float = 0.0
    formatting_score: float = 0.0
    
    def get_letter_grade(self) -> str:
        """Get letter grade based on overall score."""
        if self.overall_score >= 90:
            return "A+"
        elif self.overall_score >= 85:
            return "A"
        elif self.overall_score >= 80:
            return "A-"
        elif self.overall_score >= 75:
            return "B+"
        elif self.overall_score >= 70:
            return "B"
        elif self.overall_score >= 65:
            return "B-"
        elif self.overall_score >= 60:
            return "C+"
        elif self.overall_score >= 55:
            return "C"
        elif self.overall_score >= 50:
            return "C-"
        elif self.overall_score >= 45:
            return "D"
        else:
            return "F"
    
    def get_score_interpretation(self) -> str:
        """Get interpretation of the score."""
        if self.overall_score >= 80:
            return "Excellent Match"
        elif self.overall_score >= 65:
            return "Good Match"
        elif self.overall_score >= 50:
            return "Fair Match"
        elif self.overall_score >= 35:
            return "Needs Improvement"
        else:
            return "Poor Match"


class ATSEngine:
    """
    Advanced ATS scoring engine using TF-IDF vectorization
    and cosine similarity for content matching.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.skill_engine = SkillEngine()
        
        # TF-IDF Vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
    
    def calculate_ats_score(self, 
                            resume_text: str, 
                            jd_text: str) -> Tuple[ATSScore, Dict]:
        """
        Calculate comprehensive ATS score comparing resume to job description.
        """
        score = ATSScore()
        analysis = {}
        
        if not resume_text or not jd_text:
            return score, analysis
        
        # 1. Preprocess texts
        resume_processed = self.preprocessor.preprocess(resume_text)
        jd_processed = self.preprocessor.preprocess(jd_text)
        
        # 2. Calculate content similarity using TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_processed, jd_processed])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score.content_similarity_score = min(cosine_sim * 100, 100)
        except Exception:
            score.content_similarity_score = 0
        
        # 3. Extract and compare skills
        resume_skills_result = self.skill_engine.extract_skills(resume_text)
        jd_skills_result = self.skill_engine.extract_skills(jd_text)
        
        resume_skills = resume_skills_result.get_skills_list()
        jd_skills = jd_skills_result.get_skills_list()
        
        skill_comparison = self.skill_engine.compare_skills(resume_skills, jd_skills)
        
        # Calculate skill match score
        if jd_skills:
            skill_match_ratio = len(skill_comparison["matched"]) / len(jd_skills)
            score.skill_match_score = min(skill_match_ratio * 100, 100)
        else:
            score.skill_match_score = 50  # Neutral if no skills in JD
        
        # 4. Calculate keyword density score
        jd_keywords = set(jd_processed.split())
        resume_keywords = set(resume_processed.split())
        
        if jd_keywords:
            keyword_overlap = len(jd_keywords.intersection(resume_keywords)) / len(jd_keywords)
            score.keyword_density_score = min(keyword_overlap * 100, 100)
        
        # 5. Experience match analysis
        resume_years = TextUtils.extract_years_of_experience(resume_text)
        jd_years = TextUtils.extract_years_of_experience(jd_text)
        
        if resume_years and jd_years:
            if resume_years >= jd_years:
                score.experience_match_score = 100
            else:
                ratio = resume_years / jd_years
                score.experience_match_score = min(ratio * 100, 100)
        else:
            score.experience_match_score = 70  # Neutral if can't determine
        
        # 6. Formatting/Structure score (basic checks)
        score.formatting_score = self._calculate_formatting_score(resume_text)
        
        # 7. Calculate overall weighted score
        score.overall_score = self._calculate_weighted_score(score)
        
        # Build analysis dictionary
        analysis = {
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": skill_comparison["matched"],
            "missing_skills": skill_comparison["missing"],
            "additional_skills": skill_comparison["additional"],
            "resume_skill_categories": resume_skills_result.skill_categories,
            "jd_skill_categories": jd_skills_result.skill_categories,
            "resume_word_count": len(resume_text.split()),
            "jd_word_count": len(jd_text.split()),
            "resume_years_experience": resume_years,
            "jd_years_required": jd_years
        }
        
        return score, analysis
    
    def _calculate_formatting_score(self, text: str) -> float:
        """Calculate basic formatting/structure score."""
        score = 50  # Base score
        
        # Check for section headers
        common_sections = [
            'experience', 'education', 'skills', 'summary', 
            'objective', 'work history', 'employment', 'projects',
            'certifications', 'achievements', 'accomplishments'
        ]
        
        text_lower = text.lower()
        sections_found = sum(1 for section in common_sections if section in text_lower)
        score += min(sections_found * 5, 25)  # Up to 25 points for sections
        
        # Check for bullet points or structured content
        if '•' in text or '●' in text or '-' in text or '*' in text:
            score += 10
        
        # Check for proper length (not too short, not too long)
        word_count = len(text.split())
        if 200 <= word_count <= 1000:
            score += 15
        elif 100 <= word_count < 200 or 1000 < word_count <= 1500:
            score += 10
        
        return min(score, 100)
    
    def _calculate_weighted_score(self, score: ATSScore) -> float:
        """Calculate weighted overall score."""
        weights = {
            'skill_match': 0.35,
            'content_similarity': 0.30,
            'keyword_density': 0.20,
            'experience_match': 0.10,
            'formatting': 0.05
        }
        
        weighted_score = (
            score.skill_match_score * weights['skill_match'] +
            score.content_similarity_score * weights['content_similarity'] +
            score.keyword_density_score * weights['keyword_density'] +
            score.experience_match_score * weights['experience_match'] +
            score.formatting_score * weights['formatting']
        )
        
        return round(weighted_score, 1)


# =============================================================================
# ANALYSIS ORCHESTRATOR
# =============================================================================

@dataclass
class AnalysisResult:
    """Complete analysis result data class."""
    ats_score: ATSScore
    analysis: Dict
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    resume_metadata: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResumeAnalyzer:
    """
    Main orchestrator class that coordinates all analysis components.
    """
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.ats_engine = ATSEngine()
        self.preprocessor = TextPreprocessor()
    
    def analyze(self, resume_file, job_description: str) -> AnalysisResult:
        """
        Perform complete resume analysis against job description.
        """
        # Extract resume text
        resume_text, resume_metadata = self.pdf_parser.extract_text(resume_file)
        
        if not resume_text:
            raise ValueError("Could not extract text from resume PDF. Please ensure the PDF contains selectable text.")
        
        if not job_description.strip():
            raise ValueError("Job description cannot be empty.")
        
        # Calculate ATS score and get analysis
        ats_score, analysis = self.ats_engine.calculate_ats_score(resume_text, job_description)
        
        # Generate strengths, weaknesses, and recommendations
        strengths = self._identify_strengths(ats_score, analysis)
        weaknesses = self._identify_weaknesses(ats_score, analysis)
        recommendations = self._generate_recommendations(ats_score, analysis)
        
        return AnalysisResult(
            ats_score=ats_score,
            analysis=analysis,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            resume_metadata=resume_metadata
        )
    
    def _identify_strengths(self, score: ATSScore, analysis: Dict) -> List[str]:
        """Identify resume strengths based on analysis."""
        strengths = []
        
        # Skill-based strengths
        matched_count = len(analysis.get("matched_skills", []))
        if matched_count >= 10:
            strengths.append(f"Excellent skill alignment with {matched_count} matching skills identified")
        elif matched_count >= 5:
            strengths.append(f"Good skill coverage with {matched_count} relevant skills found")
        
        # Score-based strengths
        if score.content_similarity_score >= 70:
            strengths.append("Strong content alignment with job requirements")
        
        if score.skill_match_score >= 75:
            strengths.append("High percentage of required skills present in resume")
        
        if score.experience_match_score >= 80:
            strengths.append("Experience level aligns well with position requirements")
        
        if score.formatting_score >= 80:
            strengths.append("Well-structured resume format with clear sections")
        
        # Additional skill strengths
        additional_skills = analysis.get("additional_skills", [])
        if len(additional_skills) >= 5:
            strengths.append(f"Demonstrates {len(additional_skills)} additional valuable skills beyond requirements")
        
        # Category-based strengths
        resume_categories = analysis.get("resume_skill_categories", {})
        if len(resume_categories) >= 5:
            strengths.append(f"Diverse skill set spanning {len(resume_categories)} different categories")
        
        # Default if no strengths identified
        if not strengths:
            strengths.append("Resume submitted for analysis")
        
        return strengths
    
    def _identify_weaknesses(self, score: ATSScore, analysis: Dict) -> List[str]:
        """Identify resume weaknesses based on analysis."""
        weaknesses = []
        
        # Missing skills
        missing_skills = analysis.get("missing_skills", [])
        if len(missing_skills) >= 5:
            top_missing = missing_skills[:5]
            weaknesses.append(f"Missing {len(missing_skills)} required skills including: {', '.join(top_missing)}")
        elif missing_skills:
            weaknesses.append(f"Missing some key skills: {', '.join(missing_skills)}")
        
        # Score-based weaknesses
        if score.content_similarity_score < 50:
            weaknesses.append("Limited alignment between resume content and job description")
        
        if score.skill_match_score < 50:
            weaknesses.append("Low percentage of required skills found in resume")
        
        if score.keyword_density_score < 40:
            weaknesses.append("Resume lacks many key terms from the job description")
        
        if score.formatting_score < 60:
            weaknesses.append("Resume structure could be improved with clearer sections")
        
        # Experience gap
        resume_years = analysis.get("resume_years_experience")
        jd_years = analysis.get("jd_years_required")
        if resume_years and jd_years and resume_years < jd_years:
            gap = jd_years - resume_years
            weaknesses.append(f"Potential experience gap of {gap} year(s) compared to requirements")
        
        # Word count issues
        word_count = analysis.get("resume_word_count", 0)
        if word_count < 150:
            weaknesses.append("Resume appears too brief - consider adding more detail")
        elif word_count > 1500:
            weaknesses.append("Resume may be too lengthy - consider condensing content")
        
        return weaknesses
    
    def _generate_recommendations(self, score: ATSScore, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Skill recommendations
        missing_skills = analysis.get("missing_skills", [])
        if missing_skills:
            top_missing = missing_skills[:3]
            recommendations.append(
                f"Add missing high-priority skills: {', '.join(top_missing)}. "
                "If you have experience with these, ensure they're explicitly mentioned."
            )
        
        # Content recommendations
        if score.content_similarity_score < 60:
            recommendations.append(
                "Tailor your resume language to match the job description. "
                "Use similar terminology and phrases where applicable."
            )
        
        # Keyword recommendations
        if score.keyword_density_score < 50:
            recommendations.append(
                "Incorporate more keywords from the job posting. "
                "Focus on technical terms, tools, and methodologies mentioned."
            )
        
        # Structure recommendations
        if score.formatting_score < 70:
            recommendations.append(
                "Improve resume structure by adding clear section headers "
                "(Experience, Skills, Education, Projects) and use bullet points."
            )
        
        # Quantification recommendation
        recommendations.append(
            "Quantify achievements where possible (e.g., 'Increased efficiency by 25%', "
            "'Managed team of 5 engineers', 'Reduced costs by $50K')."
        )
        
        # ATS optimization
        if score.overall_score < 70:
            recommendations.append(
                "Consider using an ATS-friendly format: avoid tables, graphics, "
                "and unusual fonts. Stick to standard section headings."
            )
        
        # General best practices
        recommendations.append(
            "Keep your resume updated with recent projects and achievements. "
            "Ensure contact information is current and professional."
        )
        
        return recommendations


# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Professional PDF report generator using ReportLab.
    Creates polished, ATS-style analysis reports.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e')
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a4a6a')
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e'),
            borderPadding=(5, 5, 5, 5)
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=6,
            bulletIndent=10,
            leading=13
        ))
        
        # Score display style
        self.styles.add(ParagraphStyle(
            name='ScoreDisplay',
            parent=self.styles['Normal'],
            fontSize=48,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0f3460'),
            spaceAfter=10
        ))
    
    def generate_report(self, result: AnalysisResult, 
                        job_title: str = "Position") -> bytes:
        """
        Generate a comprehensive PDF report from analysis results.
        Returns PDF as bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = []
        
        # Title Section
        story.append(Paragraph("Resume Analysis Report", self.styles['CustomTitle']))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            self.styles['CustomSubtitle']
        ))
        story.append(Spacer(1, 20))
        
        # Horizontal line
        story.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#e94560'),
            spaceAfter=20
        ))
        
        # Executive Summary Section
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Score Card Table
        score = result.ats_score
        score_data = [
            ['Metric', 'Score', 'Rating'],
            ['Overall ATS Score', f'{score.overall_score:.1f}/100', score.get_letter_grade()],
            ['Skill Match', f'{score.skill_match_score:.1f}/100', self._get_rating(score.skill_match_score)],
            ['Content Similarity', f'{score.content_similarity_score:.1f}/100', self._get_rating(score.content_similarity_score)],
            ['Keyword Coverage', f'{score.keyword_density_score:.1f}/100', self._get_rating(score.keyword_density_score)],
            ['Experience Match', f'{score.experience_match_score:.1f}/100', self._get_rating(score.experience_match_score)],
        ]
        
        score_table = Table(score_data, colWidths=[200, 100, 100])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#1a1a2e')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 20))
        
        # Interpretation
        story.append(Paragraph(
            f"<b>Overall Assessment:</b> {score.get_score_interpretation()}",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 20))
        
        # Skills Analysis Section
        story.append(Paragraph("Skills Analysis", self.styles['SectionHeader']))
        
        # Matched Skills
        matched_skills = result.analysis.get("matched_skills", [])
        if matched_skills:
            story.append(Paragraph("<b>✓ Matched Skills:</b>", self.styles['BodyText']))
            skills_text = ", ".join(matched_skills[:20])
            if len(matched_skills) > 20:
                skills_text += f" ... and {len(matched_skills) - 20} more"
            story.append(Paragraph(skills_text, self.styles['BulletPoint']))
        
        story.append(Spacer(1, 10))
        
        # Missing Skills
        missing_skills = result.analysis.get("missing_skills", [])
        if missing_skills:
            story.append(Paragraph("<b>✗ Missing Skills:</b>", self.styles['BodyText']))
            skills_text = ", ".join(missing_skills[:15])
            if len(missing_skills) > 15:
                skills_text += f" ... and {len(missing_skills) - 15} more"
            story.append(Paragraph(skills_text, self.styles['BulletPoint']))
        
        story.append(Spacer(1, 10))
        
        # Additional Skills
        additional_skills = result.analysis.get("additional_skills", [])
        if additional_skills:
            story.append(Paragraph("<b>+ Additional Skills (Not Required but Valuable):</b>", self.styles['BodyText']))
            skills_text = ", ".join(additional_skills[:15])
            if len(additional_skills) > 15:
                skills_text += f" ... and {len(additional_skills) - 15} more"
            story.append(Paragraph(skills_text, self.styles['BulletPoint']))
        
        story.append(Spacer(1, 20))
        
        # Strengths Section
        story.append(Paragraph("Key Strengths", self.styles['SectionHeader']))
        for strength in result.strengths:
            story.append(Paragraph(f"✓ {strength}", self.styles['BulletPoint']))
        story.append(Spacer(1, 15))
        
        # Areas for Improvement Section
        story.append(Paragraph("Areas for Improvement", self.styles['SectionHeader']))
        for weakness in result.weaknesses:
            story.append(Paragraph(f"• {weakness}", self.styles['BulletPoint']))
        story.append(Spacer(1, 15))
        
        # Recommendations Section
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        for i, rec in enumerate(result.recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['BulletPoint']))
        
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#ddd'),
            spaceBefore=20
        ))
        
        story.append(Paragraph(
            "This report was generated by AI Resume Analyzer. "
            "Scores are calculated using advanced NLP techniques including "
            "TF-IDF vectorization and cosine similarity analysis.",
            ParagraphStyle(
                'Footer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#888'),
                alignment=TA_CENTER,
                spaceBefore=10
            )
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _get_rating(self, score: float) -> str:
        """Get text rating based on score."""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Needs Work"


# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================

def configure_page():
    """Configure Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="AI Resume Analyzer | ATS Score Checker",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme and professional styling
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #e94560;
            --secondary-color: #0f3460;
            --background-dark: #1a1a2e;
            --background-light: #16213e;
            --text-color: #eaeaea;
            --accent-color: #00d9ff;
        }
        
        /* Global styles */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #e94560, #0f3460);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0;
            padding: 20px 0 10px 0;
        }
        
        .sub-header {
            color: #a0a0a0;
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(145deg, #1e1e30, #252540);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(233, 69, 96, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        
        .score-display {
            font-size: 4rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(135deg, #e94560, #00d9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0;
        }
        
        .grade-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2rem;
            margin: 10px 0;
        }
        
        .grade-a { background: linear-gradient(135deg, #00b894, #00cec9); color: white; }
        .grade-b { background: linear-gradient(135deg, #0984e3, #74b9ff); color: white; }
        .grade-c { background: linear-gradient(135deg, #fdcb6e, #f39c12); color: white; }
        .grade-d { background: linear-gradient(135deg, #e17055, #d63031); color: white; }
        
        /* Skills styling */
        .skill-tag {
            display: inline-block;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .skill-matched {
            background: rgba(0, 184, 148, 0.2);
            color: #00b894;
            border: 1px solid #00b894;
        }
        
        .skill-missing {
            background: rgba(214, 48, 49, 0.2);
            color: #ff6b6b;
            border: 1px solid #ff6b6b;
        }
        
        .skill-additional {
            background: rgba(9, 132, 227, 0.2);
            color: #74b9ff;
            border: 1px solid #74b9ff;
        }
        
        /* Section headers */
        .section-header {
            color: #e94560;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(233, 69, 96, 0.3);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e, #16213e);
            border-right: 1px solid rgba(233, 69, 96, 0.3);
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #eaeaea;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #e94560, #0f3460);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
        }
        
        /* File uploader styling */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            border: 2px dashed rgba(233, 69, 96, 0.3);
        }
        
        /* Text area styling */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(233, 69, 96, 0.3);
            border-radius: 10px;
            color: #eaeaea;
        }
        
        /* Progress bar custom styling */
        .stProgress > div > div {
            background: linear-gradient(90deg, #e94560, #00d9ff);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        /* Alert boxes */
        .success-box {
            background: rgba(0, 184, 148, 0.1);
            border-left: 4px solid #00b894;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .warning-box {
            background: rgba(253, 203, 110, 0.1);
            border-left: 4px solid #fdcb6e;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .info-box {
            background: rgba(116, 185, 255, 0.1);
            border-left: 4px solid #74b9ff;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Stats row */
        .stats-row {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px 25px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            min-width: 120px;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d9ff;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #a0a0a0;
            margin-top: 5px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            .score-display {
                font-size: 3rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">🎯 AI Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced ATS Scoring System powered by Machine Learning & NLP</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with information and controls."""
    with st.sidebar:
        st.markdown("## 📋 How It Works")
        st.markdown("""
        1. **Upload** your resume (PDF format)
        2. **Paste** the job description
        3. **Analyze** to get your ATS score
        4. **Download** detailed PDF report
        """)
        
        st.markdown("---")
        
        st.markdown("## 🎯 Scoring Criteria")
        st.markdown("""
        - **Skill Match (35%)**: Technical & soft skills alignment
        - **Content Similarity (30%)**: Overall resume-JD match
        - **Keywords (20%)**: Important terms coverage
        - **Experience (10%)**: Years of experience match
        - **Format (5%)**: Resume structure quality
        """)
        
        st.markdown("---")
        
        st.markdown("## 📊 Score Guide")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🟢 **80-100**: Excellent")
            st.markdown("🔵 **60-79**: Good")
        with col2:
            st.markdown("🟡 **40-59**: Fair")
            st.markdown("🔴 **0-39**: Needs Work")
        
        st.markdown("---")
        
        st.markdown("## ⚡ Features")
        st.markdown("""
        - ✅ TF-IDF Analysis
        - ✅ Cosine Similarity
        - ✅ Skill Extraction
        - ✅ Gap Analysis
        - ✅ PDF Reports
        - ✅ Recommendations
        """)
        
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #666; font-size: 0.8rem;'>"
            "Built with ❤️ using Streamlit<br>"
            "© 2024 AI Resume Analyzer"
            "</p>",
            unsafe_allow_html=True
        )


def render_score_card(score: ATSScore):
    """Render the main score card with visual indicators."""
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Main score display
        st.markdown(
            f'<div class="score-display">{score.overall_score:.0f}</div>',
            unsafe_allow_html=True
        )
        
        # Letter grade badge
        grade = score.get_letter_grade()
        grade_class = "grade-a" if grade.startswith("A") else \
                      "grade-b" if grade.startswith("B") else \
                      "grade-c" if grade.startswith("C") else "grade-d"
        
        st.markdown(
            f'<div style="text-align: center;">'
            f'<span class="grade-badge {grade_class}">Grade: {grade}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            f'<p style="text-align: center; color: #a0a0a0; margin-top: 10px;">'
            f'{score.get_score_interpretation()}</p>',
            unsafe_allow_html=True
        )
        
        # Progress bar
        st.progress(score.overall_score / 100)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_detailed_scores(score: ATSScore):
    """Render detailed breakdown of all scores."""
    st.markdown('<p class="section-header">📊 Detailed Score Breakdown</p>', unsafe_allow_html=True)
    
    metrics = [
        ("Skill Match", score.skill_match_score, "🎯"),
        ("Content Similarity", score.content_similarity_score, "📝"),
        ("Keyword Coverage", score.keyword_density_score, "🔑"),
        ("Experience Match", score.experience_match_score, "💼"),
        ("Format Quality", score.formatting_score, "📋")
    ]
    
    cols = st.columns(len(metrics))
    
    for col, (name, value, icon) in zip(cols, metrics):
        with col:
            # Determine color based on score
            if value >= 70:
                color = "#00b894"
            elif value >= 50:
                color = "#fdcb6e"
            else:
                color = "#ff6b6b"
            
            st.markdown(
                f"""
                <div style="text-align: center; padding: 15px; 
                            background: rgba(255,255,255,0.05); 
                            border-radius: 10px; margin: 5px;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: {color};">
                        {value:.0f}%
                    </div>
                    <div style="font-size: 0.8rem; color: #a0a0a0; margin-top: 5px;">
                        {name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_skills_analysis(analysis: Dict):
    """Render the skills analysis section with matched and missing skills."""
    st.markdown('<p class="section-header">🔧 Skills Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Matched Skills")
        matched_skills = analysis.get("matched_skills", [])
        if matched_skills:
            skills_html = " ".join([
                f'<span class="skill-tag skill-matched">{skill}</span>'
                for skill in matched_skills
            ])
            st.markdown(skills_html, unsafe_allow_html=True)
            st.success(f"**{len(matched_skills)}** skills matched!")
        else:
            st.info("No matching skills found")
    
    with col2:
        st.markdown("### ❌ Missing Skills")
        missing_skills = analysis.get("missing_skills", [])
        if missing_skills:
            skills_html = " ".join([
                f'<span class="skill-tag skill-missing">{skill}</span>'
                for skill in missing_skills
            ])
            st.markdown(skills_html, unsafe_allow_html=True)
            st.warning(f"**{len(missing_skills)}** skills to add!")
        else:
            st.success("You have all required skills!")
    
    # Additional skills section
    additional_skills = analysis.get("additional_skills", [])
    if additional_skills:
        with st.expander("🌟 Additional Skills You Have", expanded=False):
            skills_html = " ".join([
                f'<span class="skill-tag skill-additional">{skill}</span>'
                for skill in additional_skills
            ])
            st.markdown(skills_html, unsafe_allow_html=True)
            st.info(f"You have **{len(additional_skills)}** additional valuable skills!")


def render_strengths_weaknesses(result: AnalysisResult):
    """Render strengths and weaknesses sections."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">💪 Key Strengths</p>', unsafe_allow_html=True)
        for strength in result.strengths:
            st.markdown(
                f'<div class="success-box">✓ {strength}</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown('<p class="section-header">🎯 Areas for Improvement</p>', unsafe_allow_html=True)
        for weakness in result.weaknesses:
            st.markdown(
                f'<div class="warning-box">• {weakness}</div>',
                unsafe_allow_html=True
            )


def render_recommendations(result: AnalysisResult):
    """Render recommendations section."""
    st.markdown('<p class="section-header">💡 Recommendations</p>', unsafe_allow_html=True)
    
    for i, rec in enumerate(result.recommendations, 1):
        st.markdown(
            f'<div class="info-box"><strong>Recommendation {i}:</strong> {rec}</div>',
            unsafe_allow_html=True
        )


def render_statistics(analysis: Dict, result: AnalysisResult):
    """Render quick statistics."""
    st.markdown('<p class="section-header">📈 Quick Statistics</p>', unsafe_allow_html=True)
    
    stats = [
        ("Total Skills Found", len(analysis.get("resume_skills", []))),
        ("Skills Matched", len(analysis.get("matched_skills", []))),
        ("Skills Missing", len(analysis.get("missing_skills", []))),
        ("Resume Pages", result.resume_metadata.get("pages", "N/A")),
    ]
    
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats):
        col.metric(label, value)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Configure page
    configure_page()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'pdf_report' not in st.session_state:
        st.session_state.pdf_report = None
    
    # Main content area
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        resume_file = st.file_uploader(
            "Upload your resume in PDF format",
            type=['pdf'],
            help="Please upload a PDF file with selectable text for best results"
        )
        
        if resume_file:
            st.success(f"✅ Uploaded: {resume_file.name}")
    
    with col2:
        st.markdown("### 📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Paste the complete job description including requirements, responsibilities, and qualifications...",
            help="Include as much detail as possible for better analysis"
        )
    
    # Analyze button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Resume", use_container_width=True)
    
    # Analysis logic
    if analyze_button:
        if not resume_file:
            st.error("⚠️ Please upload a resume PDF file.")
        elif not job_description.strip():
            st.error("⚠️ Please enter a job description.")
        else:
            with st.spinner("🔄 Analyzing your resume... This may take a moment."):
                try:
                    # Initialize analyzer
                    analyzer = ResumeAnalyzer()
                    
                    # Perform analysis
                    result = analyzer.analyze(resume_file, job_description)
                    
                    # Store in session state
                    st.session_state.analysis_result = result
                    
                    # Generate PDF report
                    report_generator = ReportGenerator()
                    pdf_bytes = report_generator.generate_report(result)
                    st.session_state.pdf_report = pdf_bytes
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
    
    # Display results if available
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Main score card
        render_score_card(result.ats_score)
        
        # Detailed scores
        render_detailed_scores(result.ats_score)
        
        st.markdown("---")
        
        # Statistics
        render_statistics(result.analysis, result)
        
        st.markdown("---")
        
        # Skills analysis
        render_skills_analysis(result.analysis)
        
        st.markdown("---")
        
        # Strengths and weaknesses
        render_strengths_weaknesses(result)
        
        st.markdown("---")
        
        # Recommendations
        render_recommendations(result)
        
        st.markdown("---")
        
        # Download section
        st.markdown("### 📥 Download Report")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.pdf_report:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=st.session_state.pdf_report,
                    file_name=f"resume_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # Additional analysis details (expandable)
        with st.expander("🔬 Technical Details", expanded=False):
            st.markdown("#### Analysis Metadata")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Resume Statistics:**")
                st.write(f"- Word Count: {result.analysis.get('resume_word_count', 'N/A')}")
                st.write(f"- Pages: {result.resume_metadata.get('pages', 'N/A')}")
                st.write(f"- Extraction Method: {result.resume_metadata.get('extraction_method', 'N/A')}")
                years_exp = result.analysis.get('resume_years_experience')
                st.write(f"- Years of Experience Detected: {years_exp if years_exp else 'Not detected'}")
            
            with col2:
                st.write("**Job Description Statistics:**")
                st.write(f"- Word Count: {result.analysis.get('jd_word_count', 'N/A')}")
                st.write(f"- Skills Required: {len(result.analysis.get('jd_skills', []))}")
                jd_years = result.analysis.get('jd_years_required')
                st.write(f"- Experience Required: {jd_years if jd_years else 'Not specified'}")
            
            st.markdown("#### Skill Categories in Resume")
            categories = result.analysis.get("resume_skill_categories", {})
            if categories:
                for category, skills in categories.items():
                    st.write(f"**{category}:** {', '.join(skills[:10])}")
            
            st.markdown("#### Analysis Timestamp")
            st.write(f"Generated at: {result.timestamp}")


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
