"""
AI Resume Analyzer - Production-Grade ATS System
================================================
A comprehensive resume analysis tool that matches resumes against job descriptions
using NLP techniques including TF-IDF vectorization and cosine similarity scoring.

Author: Senior AI Engineer
Version: 1.0.0
License: MIT
"""

import streamlit as st
import PyPDF2
import io
import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus import HRFlowable, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import datetime
import base64
from dataclasses import dataclass
from enum import Enum
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration settings."""
    APP_TITLE = "AI Resume Analyzer"
    APP_ICON = "📄"
    APP_DESCRIPTION = "Professional ATS-Style Resume Analysis System"
    VERSION = "1.0.0"
    
    # Scoring weights
    SKILL_MATCH_WEIGHT = 0.4
    CONTENT_SIMILARITY_WEIGHT = 0.35
    KEYWORD_DENSITY_WEIGHT = 0.25
    
    # Thresholds
    EXCELLENT_SCORE = 80
    GOOD_SCORE = 60
    FAIR_SCORE = 40
    
    # TF-IDF settings
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    
    # Report settings
    COMPANY_NAME = "AI Resume Analyzer"
    REPORT_AUTHOR = "ATS Analysis Engine"


# ============================================================================
# SKILLS DATABASE
# ============================================================================

class SkillsDatabase:
    """
    Comprehensive skills database organized by category.
    Contains 500+ skills across multiple domains for accurate matching.
    """
    
    PROGRAMMING_LANGUAGES = {
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "ruby", "php", "swift", "kotlin", "go", "golang", "rust", "scala",
        "perl", "r", "matlab", "julia", "dart", "lua", "haskell", "erlang",
        "clojure", "elixir", "f#", "visual basic", "vba", "cobol", "fortran",
        "assembly", "groovy", "objective-c", "shell", "bash", "powershell",
        "sql", "plsql", "tsql", "nosql", "graphql"
    }
    
    WEB_TECHNOLOGIES = {
        "html", "html5", "css", "css3", "sass", "scss", "less", "bootstrap",
        "tailwind", "tailwindcss", "jquery", "react", "reactjs", "react.js",
        "angular", "angularjs", "vue", "vuejs", "vue.js", "svelte", "nextjs",
        "next.js", "nuxt", "nuxtjs", "gatsby", "webpack", "babel", "npm",
        "yarn", "pnpm", "vite", "rollup", "parcel", "gulp", "grunt",
        "express", "expressjs", "nodejs", "node.js", "node", "deno", "bun",
        "django", "flask", "fastapi", "pyramid", "tornado", "aiohttp",
        "spring", "spring boot", "springboot", "hibernate", "struts",
        "asp.net", "asp.net core", ".net", ".net core", "blazor",
        "ruby on rails", "rails", "sinatra", "laravel", "symfony", "codeigniter",
        "rest", "restful", "rest api", "soap", "grpc", "websocket", "ajax",
        "json", "xml", "yaml", "jwt", "oauth", "oauth2", "openid"
    }
    
    DATABASES = {
        "mysql", "postgresql", "postgres", "sqlite", "oracle", "sql server",
        "mssql", "mariadb", "mongodb", "cassandra", "couchdb", "couchbase",
        "dynamodb", "redis", "memcached", "elasticsearch", "solr",
        "neo4j", "arangodb", "firebase", "firestore", "supabase",
        "cockroachdb", "timescaledb", "influxdb", "clickhouse", "snowflake",
        "bigquery", "redshift", "data warehouse", "data lake", "etl",
        "database design", "database optimization", "indexing", "sharding",
        "replication", "acid", "cap theorem"
    }
    
    CLOUD_PLATFORMS = {
        "aws", "amazon web services", "azure", "microsoft azure", "gcp",
        "google cloud", "google cloud platform", "heroku", "digitalocean",
        "linode", "vultr", "oracle cloud", "ibm cloud", "alibaba cloud",
        "ec2", "s3", "lambda", "cloudfront", "route53", "rds", "dynamodb",
        "sqs", "sns", "kinesis", "emr", "redshift", "athena", "glue",
        "cloudformation", "cdk", "sam", "elastic beanstalk", "ecs", "eks",
        "fargate", "app runner", "lightsail", "amplify",
        "azure functions", "azure devops", "azure sql", "cosmos db",
        "cloud functions", "cloud run", "app engine", "compute engine",
        "cloud storage", "bigquery", "dataflow", "pubsub", "vertex ai"
    }
    
    DEVOPS_TOOLS = {
        "docker", "kubernetes", "k8s", "openshift", "rancher", "helm",
        "terraform", "ansible", "puppet", "chef", "saltstack", "vagrant",
        "jenkins", "gitlab ci", "github actions", "circleci", "travis ci",
        "bamboo", "teamcity", "azure pipelines", "argocd", "flux",
        "prometheus", "grafana", "datadog", "new relic", "splunk",
        "elk stack", "elasticsearch", "logstash", "kibana", "fluentd",
        "nagios", "zabbix", "pagerduty", "opsgenie",
        "nginx", "apache", "haproxy", "traefik", "envoy", "istio",
        "consul", "vault", "etcd", "zookeeper",
        "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
        "ci/cd", "continuous integration", "continuous deployment",
        "infrastructure as code", "iac", "gitops", "devsecops", "sre"
    }
    
    DATA_SCIENCE_ML = {
        "machine learning", "deep learning", "artificial intelligence", "ai",
        "neural networks", "natural language processing", "nlp",
        "computer vision", "image processing", "speech recognition",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "tensorflow", "keras", "pytorch", "torch", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "random forest", "decision tree",
        "gradient boosting", "svm", "support vector machine", "naive bayes",
        "linear regression", "logistic regression", "clustering", "k-means",
        "pca", "dimensionality reduction", "feature engineering",
        "model training", "hyperparameter tuning", "cross-validation",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "jupyter", "jupyter notebook", "google colab", "kaggle",
        "hugging face", "transformers", "bert", "gpt", "llm", "langchain",
        "openai", "stable diffusion", "generative ai", "rag",
        "mlops", "mlflow", "kubeflow", "sagemaker", "vertex ai",
        "data analysis", "data visualization", "statistical analysis",
        "a/b testing", "hypothesis testing", "time series", "forecasting",
        "recommendation systems", "anomaly detection", "sentiment analysis"
    }
    
    MOBILE_DEVELOPMENT = {
        "ios", "android", "react native", "flutter", "xamarin", "ionic",
        "cordova", "phonegap", "swift", "swiftui", "uikit", "objective-c",
        "kotlin", "java android", "jetpack compose", "android studio",
        "xcode", "cocoapods", "gradle", "mobile app development",
        "app store", "google play", "push notifications", "mobile ui/ux",
        "responsive design", "pwa", "progressive web app"
    }
    
    TESTING = {
        "unit testing", "integration testing", "e2e testing", "end-to-end testing",
        "test automation", "selenium", "cypress", "playwright", "puppeteer",
        "jest", "mocha", "chai", "jasmine", "karma", "pytest", "unittest",
        "junit", "testng", "mockito", "rspec", "capybara",
        "postman", "insomnia", "api testing", "load testing", "jmeter",
        "gatling", "locust", "performance testing", "stress testing",
        "tdd", "test-driven development", "bdd", "behavior-driven development",
        "qa", "quality assurance", "manual testing", "regression testing",
        "smoke testing", "sanity testing", "uat", "user acceptance testing"
    }
    
    SOFT_SKILLS = {
        "communication", "leadership", "teamwork", "collaboration",
        "problem solving", "problem-solving", "critical thinking",
        "analytical skills", "creativity", "innovation", "adaptability",
        "flexibility", "time management", "project management",
        "agile", "scrum", "kanban", "waterfall", "jira", "confluence",
        "trello", "asana", "monday", "notion", "slack",
        "presentation", "public speaking", "negotiation", "mentoring",
        "coaching", "decision making", "strategic thinking", "planning",
        "organization", "attention to detail", "multitasking",
        "customer service", "client relations", "stakeholder management",
        "cross-functional", "remote work", "self-motivated", "proactive"
    }
    
    SECURITY = {
        "cybersecurity", "information security", "network security",
        "application security", "cloud security", "devsecops",
        "penetration testing", "ethical hacking", "vulnerability assessment",
        "siem", "soc", "incident response", "threat modeling",
        "encryption", "ssl", "tls", "https", "oauth", "saml", "sso",
        "identity management", "iam", "rbac", "access control",
        "firewall", "ids", "ips", "vpn", "zero trust",
        "owasp", "security auditing", "compliance", "gdpr", "hipaa", "pci-dss",
        "soc2", "iso 27001", "nist", "risk assessment"
    }
    
    DESIGN_TOOLS = {
        "figma", "sketch", "adobe xd", "invision", "zeplin", "framer",
        "photoshop", "illustrator", "after effects", "premiere pro",
        "indesign", "lightroom", "canva", "gimp", "inkscape", "blender",
        "ui design", "ux design", "ui/ux", "user interface", "user experience",
        "wireframing", "prototyping", "mockups", "design systems",
        "typography", "color theory", "visual design", "graphic design",
        "responsive design", "accessibility", "wcag", "a11y"
    }
    
    BUSINESS_TOOLS = {
        "excel", "microsoft excel", "google sheets", "powerpoint",
        "microsoft word", "google docs", "google slides",
        "tableau", "power bi", "looker", "metabase", "superset",
        "salesforce", "hubspot", "zendesk", "freshdesk", "intercom",
        "sap", "oracle erp", "workday", "quickbooks", "netsuite",
        "google analytics", "adobe analytics", "mixpanel", "amplitude",
        "segment", "hotjar", "google tag manager", "data studio",
        "marketing automation", "crm", "erp", "business intelligence"
    }
    
    ARCHITECTURE = {
        "microservices", "monolithic", "serverless", "event-driven",
        "domain-driven design", "ddd", "clean architecture", "hexagonal",
        "mvc", "mvvm", "mvp", "solid principles", "design patterns",
        "singleton", "factory", "observer", "strategy", "decorator",
        "api design", "api gateway", "service mesh", "message queue",
        "rabbitmq", "kafka", "apache kafka", "activemq", "celery",
        "distributed systems", "high availability", "scalability",
        "load balancing", "caching", "cdn", "performance optimization",
        "system design", "technical architecture", "solution architecture"
    }
    
    @classmethod
    def get_all_skills(cls) -> Set[str]:
        """Returns a set of all skills across all categories."""
        all_skills = set()
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, set):
                all_skills.update(attr)
        return all_skills
    
    @classmethod
    def get_skill_category(cls, skill: str) -> Optional[str]:
        """Returns the category of a given skill."""
        skill_lower = skill.lower()
        categories = {
            "Programming Languages": cls.PROGRAMMING_LANGUAGES,
            "Web Technologies": cls.WEB_TECHNOLOGIES,
            "Databases": cls.DATABASES,
            "Cloud Platforms": cls.CLOUD_PLATFORMS,
            "DevOps Tools": cls.DEVOPS_TOOLS,
            "Data Science & ML": cls.DATA_SCIENCE_ML,
            "Mobile Development": cls.MOBILE_DEVELOPMENT,
            "Testing": cls.TESTING,
            "Soft Skills": cls.SOFT_SKILLS,
            "Security": cls.SECURITY,
            "Design Tools": cls.DESIGN_TOOLS,
            "Business Tools": cls.BUSINESS_TOOLS,
            "Architecture": cls.ARCHITECTURE
        }
        for category, skills in categories.items():
            if skill_lower in skills:
                return category
        return None


# ============================================================================
# TEXT PROCESSING UTILITIES
# ============================================================================

class TextProcessor:
    """
    Handles all text extraction and preprocessing operations.
    Implements NLP best practices for document analysis.
    """
    
    def __init__(self):
        """Initialize the text processor with required NLTK resources."""
        self._ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add custom stop words relevant to resumes
        self.custom_stop_words = {
            'resume', 'cv', 'curriculum', 'vitae', 'page', 'phone', 'email',
            'address', 'linkedin', 'github', 'portfolio', 'objective',
            'summary', 'experience', 'education', 'skills', 'references',
            'available', 'upon', 'request', 'jan', 'feb', 'mar', 'apr',
            'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'january', 'february', 'march', 'april', 'june', 'july',
            'august', 'september', 'october', 'november', 'december'
        }
        self.stop_words.update(self.custom_stop_words)
    
    @staticmethod
    def _ensure_nltk_resources():
        """Download required NLTK resources if not present."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text content as string
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    logging.warning(f"Could not extract text from page {page_num}: {e}")
                    continue
            
            return ' '.join(text_content)
        except Exception as e:
            logging.error(f"PDF extraction failed: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}', '', text)
        
        # Keep alphanumeric, spaces, and some special chars for skills like C++, C#
        text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and apply lemmatization.
        
        Args:
            text: Cleaned text input
            
        Returns:
            List of lemmatized tokens
        """
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        lemmatized_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                if not token.isdigit():
                    lemmatized_token = self.lemmatizer.lemmatize(token)
                    lemmatized_tokens.append(lemmatized_token)
        
        return lemmatized_tokens
    
    def preprocess_for_analysis(self, text: str) -> str:
        """
        Full preprocessing pipeline for text analysis.
        
        Args:
            text: Raw text input
            
        Returns:
            Preprocessed text ready for analysis
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)
    
    def extract_keywords(self, text: str, top_n: int = 50) -> List[Tuple[str, int]]:
        """
        Extract most frequent keywords from text.
        
        Args:
            text: Preprocessed text
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        word_freq = Counter(tokens)
        return word_freq.most_common(top_n)


# ============================================================================
# SKILL EXTRACTION ENGINE
# ============================================================================

class SkillExtractor:
    """
    Extracts and matches skills from text using the skills database.
    Implements fuzzy matching and context-aware skill detection.
    """
    
    def __init__(self):
        """Initialize the skill extractor with the skills database."""
        self.all_skills = SkillsDatabase.get_all_skills()
        self.skill_patterns = self._build_skill_patterns()
    
    def _build_skill_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for skill matching."""
        patterns = {}
        for skill in self.all_skills:
            # Create pattern that matches the skill as a whole word
            escaped_skill = re.escape(skill)
            pattern = re.compile(r'\b' + escaped_skill + r'\b', re.IGNORECASE)
            patterns[skill] = pattern
        return patterns
    
    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract all matching skills from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of found skills
        """
        if not text:
            return set()
        
        found_skills = set()
        text_lower = text.lower()
        
        # Direct matching
        for skill, pattern in self.skill_patterns.items():
            if pattern.search(text_lower):
                found_skills.add(skill)
        
        # Additional pattern matching for compound skills
        found_skills.update(self._extract_compound_skills(text_lower))
        
        return found_skills
    
    def _extract_compound_skills(self, text: str) -> Set[str]:
        """Extract skills that might appear in different formats."""
        found = set()
        
        # Common variations
        variations = {
            'react.js': 'reactjs',
            'react js': 'reactjs',
            'vue.js': 'vuejs',
            'vue js': 'vuejs',
            'node.js': 'nodejs',
            'node js': 'nodejs',
            'next.js': 'nextjs',
            'next js': 'nextjs',
            'express.js': 'expressjs',
            'machine-learning': 'machine learning',
            'deep-learning': 'deep learning',
            'ci/cd': 'ci/cd',
            'ci cd': 'ci/cd',
            '.net core': '.net core',
            'dot net': '.net',
        }
        
        for variant, canonical in variations.items():
            if variant in text:
                found.add(canonical)
        
        return found
    
    def categorize_skills(self, skills: Set[str]) -> Dict[str, List[str]]:
        """
        Categorize skills by their domain.
        
        Args:
            skills: Set of skills to categorize
            
        Returns:
            Dictionary mapping categories to skill lists
        """
        categorized = {}
        for skill in skills:
            category = SkillsDatabase.get_skill_category(skill)
            if category:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(skill)
        
        # Sort skills within each category
        for category in categorized:
            categorized[category].sort()
        
        return categorized


# ============================================================================
# ATS MATCHING ENGINE
# ============================================================================

@dataclass
class AnalysisResult:
    """Data class to hold analysis results."""
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


class ATSEngine:
    """
    Core ATS matching engine using TF-IDF and cosine similarity.
    Implements industry-standard resume scoring algorithms.
    """
    
    def __init__(self):
        """Initialize the ATS engine with required components."""
        self.text_processor = TextProcessor()
        self.skill_extractor = SkillExtractor()
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words='english',
            lowercase=True,
            norm='l2'
        )
    
    def analyze(self, resume_text: str, job_description: str) -> AnalysisResult:
        """
        Perform comprehensive resume analysis against job description.
        
        Args:
            resume_text: Extracted resume text
            job_description: Job description text
            
        Returns:
            AnalysisResult object with all scores and insights
        """
        # Extract skills from both documents
        resume_skills = self.skill_extractor.extract_skills(resume_text)
        jd_skills = self.skill_extractor.extract_skills(job_description)
        
        # Calculate skill matches
        matched_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills - resume_skills
        resume_only_skills = resume_skills - jd_skills
        
        # Calculate skill match score
        skill_match_score = self._calculate_skill_match_score(matched_skills, jd_skills)
        
        # Calculate content similarity using TF-IDF
        content_similarity_score = self._calculate_content_similarity(resume_text, job_description)
        
        # Calculate keyword match score
        keyword_match_score, matched_keywords = self._calculate_keyword_match(resume_text, job_description)
        
        # Calculate overall score with weighted components
        overall_score = (
            skill_match_score * Config.SKILL_MATCH_WEIGHT +
            content_similarity_score * Config.CONTENT_SIMILARITY_WEIGHT +
            keyword_match_score * Config.KEYWORD_DENSITY_WEIGHT
        )
        
        # Generate insights
        strengths = self._generate_strengths(matched_skills, content_similarity_score, resume_skills)
        weaknesses = self._generate_weaknesses(missing_skills, overall_score)
        recommendations = self._generate_recommendations(missing_skills, overall_score, matched_skills)
        
        # Categorize skills
        skill_categories = self.skill_extractor.categorize_skills(matched_skills)
        missing_categories = self.skill_extractor.categorize_skills(missing_skills)
        
        return AnalysisResult(
            overall_score=round(overall_score, 1),
            skill_match_score=round(skill_match_score, 1),
            content_similarity_score=round(content_similarity_score, 1),
            keyword_match_score=round(keyword_match_score, 1),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            resume_only_skills=resume_only_skills,
            matched_keywords=matched_keywords,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            skill_categories=skill_categories,
            missing_categories=missing_categories
        )
    
    def _calculate_skill_match_score(self, matched: Set[str], required: Set[str]) -> float:
        """Calculate the skill match percentage score."""
        if not required:
            return 100.0
        return (len(matched) / len(required)) * 100
    
    def _calculate_content_similarity(self, resume: str, job_desc: str) -> float:
        """Calculate TF-IDF based cosine similarity score."""
        try:
            # Preprocess texts
            resume_processed = self.text_processor.preprocess_for_analysis(resume)
            jd_processed = self.text_processor.preprocess_for_analysis(job_desc)
            
            if not resume_processed or not jd_processed:
                return 0.0
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([resume_processed, jd_processed])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Convert to percentage (0-100)
            return similarity * 100
        except Exception as e:
            logging.error(f"Content similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_keyword_match(self, resume: str, job_desc: str) -> Tuple[float, List[str]]:
        """Calculate keyword matching score and return matched keywords."""
        jd_keywords = self.text_processor.extract_keywords(job_desc, top_n=30)
        resume_lower = resume.lower()
        
        matched_keywords = []
        for keyword, _ in jd_keywords:
            if keyword in resume_lower:
                matched_keywords.append(keyword)
        
        if not jd_keywords:
            return 100.0, []
        
        score = (len(matched_keywords) / len(jd_keywords)) * 100
        return score, matched_keywords[:15]  # Return top 15 matched keywords
    
    def _generate_strengths(self, matched_skills: Set[str], 
                           similarity_score: float, 
                           all_resume_skills: Set[str]) -> List[str]:
        """Generate list of strengths based on analysis."""
        strengths = []
        
        if len(matched_skills) >= 10:
            strengths.append(f"Strong skill alignment with {len(matched_skills)} matching skills")
        elif len(matched_skills) >= 5:
            strengths.append(f"Good skill coverage with {len(matched_skills)} matching skills")
        
        if similarity_score >= 70:
            strengths.append("Excellent content relevance to job description")
        elif similarity_score >= 50:
            strengths.append("Good content alignment with job requirements")
        
        if len(all_resume_skills) >= 20:
            strengths.append(f"Diverse skill portfolio with {len(all_resume_skills)} total skills")
        
        # Check for high-demand skills
        high_demand = {'python', 'javascript', 'aws', 'docker', 'kubernetes', 'machine learning'}
        found_high_demand = matched_skills.intersection(high_demand)
        if found_high_demand:
            strengths.append(f"Possesses high-demand skills: {', '.join(list(found_high_demand)[:3])}")
        
        if not strengths:
            strengths.append("Resume contains relevant professional content")
        
        return strengths
    
    def _generate_weaknesses(self, missing_skills: Set[str], overall_score: float) -> List[str]:
        """Generate list of weaknesses based on analysis."""
        weaknesses = []
        
        if len(missing_skills) > 10:
            weaknesses.append(f"Missing {len(missing_skills)} skills mentioned in job description")
        elif len(missing_skills) > 5:
            weaknesses.append(f"Could improve coverage on {len(missing_skills)} required skills")
        
        if overall_score < Config.FAIR_SCORE:
            weaknesses.append("Overall alignment with job requirements needs improvement")
        
        # Identify critical missing skills
        critical_skills = {'python', 'java', 'javascript', 'sql', 'aws', 'docker'}
        missing_critical = missing_skills.intersection(critical_skills)
        if missing_critical:
            weaknesses.append(f"Missing commonly required skills: {', '.join(list(missing_critical)[:3])}")
        
        if not weaknesses:
            if missing_skills:
                weaknesses.append("Minor gaps in skill coverage")
            else:
                weaknesses.append("No significant weaknesses identified")
        
        return weaknesses
    
    def _generate_recommendations(self, missing_skills: Set[str], 
                                  overall_score: float,
                                  matched_skills: Set[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if missing_skills:
            # Prioritize missing skills by importance
            priority_missing = []
            for skill in missing_skills:
                category = SkillsDatabase.get_skill_category(skill)
                if category in ['Programming Languages', 'Cloud Platforms', 'Data Science & ML']:
                    priority_missing.append(skill)
            
            if priority_missing:
                top_skills = list(priority_missing)[:5]
                recommendations.append(f"Consider adding these key skills: {', '.join(top_skills)}")
        
        if overall_score < Config.GOOD_SCORE:
            recommendations.append("Tailor resume content more closely to the job description")
            recommendations.append("Use keywords from the job posting in your experience section")
        
        if len(matched_skills) < 5:
            recommendations.append("Expand skills section with more relevant technical skills")
        
        recommendations.append("Quantify achievements with metrics where possible")
        recommendations.append("Ensure consistent formatting and clear section headers")
        
        return recommendations[:6]  # Return top 6 recommendations


# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """
    Generates professional PDF reports for analysis results.
    Uses ReportLab for high-quality document generation.
    """
    
    def __init__(self):
        """Initialize the report generator with styles."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e'),
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e'),
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#0f3460')
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=4,
            spaceAfter=4,
            alignment=TA_JUSTIFY
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceBefore=2,
            spaceAfter=2
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            parent=self.styles['Normal'],
            fontSize=36,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=10
        ))
    
    def generate_report(self, result: AnalysisResult, resume_name: str = "Resume") -> bytes:
        """
        Generate a comprehensive PDF report.
        
        Args:
            result: AnalysisResult object with analysis data
            resume_name: Name of the analyzed resume file
            
        Returns:
            PDF file content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = []
        
        # Add title
        story.append(Paragraph("Resume Analysis Report", self.styles['ReportTitle']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
        story.append(Spacer(1, 20))
        
        # Add metadata
        date_str = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(f"<b>Analysis Date:</b> {date_str}", self.styles['BodyText']))
        story.append(Paragraph(f"<b>Document:</b> {resume_name}", self.styles['BodyText']))
        story.append(Paragraph(f"<b>Generated by:</b> {Config.COMPANY_NAME} v{Config.VERSION}", self.styles['BodyText']))
        story.append(Spacer(1, 30))
        
        # Add overall score section
        story.append(Paragraph("Overall ATS Score", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        
        # Score display with color
        score_color = self._get_score_color(result.overall_score)
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<font color='{score_color}'><b>{result.overall_score}/100</b></font>",
            self.styles['ScoreText']
        ))
        story.append(Paragraph(
            f"<font color='{score_color}'>{self._get_score_label(result.overall_score)}</font>",
            ParagraphStyle('ScoreLabel', parent=self.styles['Normal'], 
                          alignment=TA_CENTER, fontSize=14)
        ))
        story.append(Spacer(1, 20))
        
        # Score breakdown table
        score_data = [
            ['Component', 'Score', 'Weight'],
            ['Skill Match', f"{result.skill_match_score}%", f"{int(Config.SKILL_MATCH_WEIGHT * 100)}%"],
            ['Content Similarity', f"{result.content_similarity_score}%", f"{int(Config.CONTENT_SIMILARITY_WEIGHT * 100)}%"],
            ['Keyword Match', f"{result.keyword_match_score}%", f"{int(Config.KEYWORD_DENSITY_WEIGHT * 100)}%"],
        ]
        
        score_table = Table(score_data, colWidths=[200, 100, 100])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 30))
        
        # Matched Skills Section
        story.append(Paragraph("Matched Skills", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 10))
        
        if result.matched_skills:
            story.append(Paragraph(
                f"<b>{len(result.matched_skills)}</b> skills from the job description found in your resume:",
                self.styles['BodyText']
            ))
            story.append(Spacer(1, 5))
            
            # Skills by category
            for category, skills in result.skill_categories.items():
                story.append(Paragraph(f"<b>{category}:</b>", self.styles['SubsectionHeader']))
                skills_text = ", ".join(sorted(skills))
                story.append(Paragraph(f"• {skills_text}", self.styles['BulletPoint']))
        else:
            story.append(Paragraph("No matching skills found.", self.styles['BodyText']))
        story.append(Spacer(1, 20))
        
        # Missing Skills Section
        story.append(Paragraph("Missing Skills", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 10))
        
        if result.missing_skills:
            story.append(Paragraph(
                f"<b>{len(result.missing_skills)}</b> skills from the job description not found in your resume:",
                self.styles['BodyText']
            ))
            story.append(Spacer(1, 5))
            
            for category, skills in result.missing_categories.items():
                story.append(Paragraph(f"<b>{category}:</b>", self.styles['SubsectionHeader']))
                skills_text = ", ".join(sorted(skills))
                story.append(Paragraph(f"• {skills_text}", self.styles['BulletPoint']))
        else:
            story.append(Paragraph("All required skills are present in your resume!", self.styles['BodyText']))
        story.append(Spacer(1, 20))
        
        # Strengths Section
        story.append(Paragraph("Strengths", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 10))
        
        for strength in result.strengths:
            story.append(Paragraph(f"✓ {strength}", self.styles['BulletPoint']))
        story.append(Spacer(1, 20))
        
        # Weaknesses Section
        story.append(Paragraph("Areas for Improvement", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 10))
        
        for weakness in result.weaknesses:
            story.append(Paragraph(f"⚠ {weakness}", self.styles['BulletPoint']))
        story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 10))
        
        for i, rec in enumerate(result.recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['BulletPoint']))
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"This report was automatically generated by {Config.COMPANY_NAME}. "
            f"For best results, tailor your resume to each specific job application.",
            ParagraphStyle('Footer', parent=self.styles['Normal'], 
                          fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
        ))
        
        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value."""
        if score >= Config.EXCELLENT_SCORE:
            return '#28a745'  # Green
        elif score >= Config.GOOD_SCORE:
            return '#5cb85c'  # Light green
        elif score >= Config.FAIR_SCORE:
            return '#ffc107'  # Yellow
        else:
            return '#dc3545'  # Red
    
    def _get_score_label(self, score: float) -> str:
        """Get label based on score value."""
        if score >= Config.EXCELLENT_SCORE:
            return 'Excellent Match'
        elif score >= Config.GOOD_SCORE:
            return 'Good Match'
        elif score >= Config.FAIR_SCORE:
            return 'Fair Match'
        else:
            return 'Needs Improvement'


# ============================================================================
# STREAMLIT UI
# ============================================================================

def setup_page():
    """Configure Streamlit page settings and apply custom styling."""
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme and professional styling
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #0e1117;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .main-header h1 {
            color: #ffffff;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .main-header p {
            color: #a0a0a0;
            font-size: 1.1rem;
        }
        
        /* Score display */
        .score-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .score-value {
            font-size: 4rem;
            font-weight: bold;
            margin: 0;
        }
        
        .score-label {
            font-size: 1.2rem;
            color: #a0a0a0;
            margin-top: 0.5rem;
        }
        
        /* Card styling */
        .metric-card {
            background-color: #1a1a2e;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        .metric-card h3 {
            color: #ffffff;
            margin-bottom: 0.5rem;
        }
        
        .metric-card p {
            color: #a0a0a0;
            margin: 0;
        }
        
        /* Skills tags */
        .skill-tag {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            margin: 0.2rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .skill-matched {
            background-color: #28a74533;
            color: #28a745;
            border: 1px solid #28a745;
        }
        
        .skill-missing {
            background-color: #dc354533;
            color: #dc3545;
            border: 1px solid #dc3545;
        }
        
        .skill-extra {
            background-color: #17a2b833;
            color: #17a2b8;
            border: 1px solid #17a2b8;
        }
        
        /* Section styling */
        .section-header {
            color: #ffffff;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #16213e;
        }
        
        /* Insight cards */
        .insight-card {
            background-color: #16213e;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid;
        }
        
        .insight-strength {
            border-left-color: #28a745;
        }
        
        .insight-weakness {
            border-left-color: #ffc107;
        }
        
        .insight-recommendation {
            border-left-color: #17a2b8;
        }
        
        /* Progress bar customization */
        .stProgress > div > div > div > div {
            background-color: #28a745;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        /* File uploader styling */
        .stFileUploader {
            background-color: #1a1a2e;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            background-color: #1a1a2e;
            color: #ffffff;
            border: 1px solid #16213e;
            border-radius: 8px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #1a1a2e;
            border-radius: 8px;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #0e1117;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom divider */
        .custom-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #16213e, transparent);
            margin: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>📄 AI Resume Analyzer</h1>
        <p>Professional ATS-Style Resume Analysis & Optimization System</p>
    </div>
    """, unsafe_allow_html=True)


def get_score_color(score: float) -> str:
    """Get color hex code based on score value."""
    if score >= Config.EXCELLENT_SCORE:
        return '#28a745'
    elif score >= Config.GOOD_SCORE:
        return '#5cb85c'
    elif score >= Config.FAIR_SCORE:
        return '#ffc107'
    else:
        return '#dc3545'


def get_score_label(score: float) -> str:
    """Get descriptive label based on score value."""
    if score >= Config.EXCELLENT_SCORE:
        return 'Excellent Match'
    elif score >= Config.GOOD_SCORE:
        return 'Good Match'
    elif score >= Config.FAIR_SCORE:
        return 'Fair Match'
    else:
        return 'Needs Improvement'


def display_score_section(result: AnalysisResult):
    """Display the main score section with visual elements."""
    score_color = get_score_color(result.overall_score)
    score_label = get_score_label(result.overall_score)
    
    st.markdown(f"""
    <div class="score-container">
        <p class="score-value" style="color: {score_color};">{result.overall_score}</p>
        <p class="score-label">{score_label}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    st.progress(result.overall_score / 100)
    
    # Score breakdown
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {get_score_color(result.skill_match_score)};">
                {result.skill_match_score}%
            </h3>
            <p>Skill Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {get_score_color(result.content_similarity_score)};">
                {result.content_similarity_score}%
            </h3>
            <p>Content Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {get_score_color(result.keyword_match_score)};">
                {result.keyword_match_score}%
            </h3>
            <p>Keyword Match</p>
        </div>
        """, unsafe_allow_html=True)


def display_skills_section(result: AnalysisResult):
    """Display matched and missing skills in a two-column layout."""
    st.markdown('<p class="section-header">📊 Skills Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Matched Skills")
        if result.matched_skills:
            skills_html = ""
            for skill in sorted(result.matched_skills):
                skills_html += f'<span class="skill-tag skill-matched">{skill}</span>'
            st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No matching skills found")
        
        # Show by category
        if result.skill_categories:
            with st.expander("View by Category"):
                for category, skills in result.skill_categories.items():
                    st.markdown(f"**{category}**")
                    st.write(", ".join(sorted(skills)))
    
    with col2:
        st.markdown("### ❌ Missing Skills")
        if result.missing_skills:
            skills_html = ""
            for skill in sorted(result.missing_skills):
                skills_html += f'<span class="skill-tag skill-missing">{skill}</span>'
            st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
        else:
            st.success("All required skills are present!")
        
        # Show by category
        if result.missing_categories:
            with st.expander("View by Category"):
                for category, skills in result.missing_categories.items():
                    st.markdown(f"**{category}**")
                    st.write(", ".join(sorted(skills)))
    
    # Additional skills in resume
    if result.resume_only_skills:
        st.markdown("### 🌟 Additional Skills (In Resume Only)")
        skills_html = ""
        for skill in sorted(list(result.resume_only_skills)[:20]):
            skills_html += f'<span class="skill-tag skill-extra">{skill}</span>'
        st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)


def display_insights_section(result: AnalysisResult):
    """Display strengths, weaknesses, and recommendations."""
    st.markdown('<p class="section-header">💡 Insights & Recommendations</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💪 Strengths")
        for strength in result.strengths:
            st.markdown(f"""
            <div class="insight-card insight-strength">
                ✓ {strength}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚠️ Areas for Improvement")
        for weakness in result.weaknesses:
            st.markdown(f"""
            <div class="insight-card insight-weakness">
                ⚡ {weakness}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Recommendations")
    for i, rec in enumerate(result.recommendations, 1):
        st.markdown(f"""
        <div class="insight-card insight-recommendation">
            {i}. {rec}
        </div>
        """, unsafe_allow_html=True)


def display_keywords_section(result: AnalysisResult):
    """Display matched keywords from job description."""
    if result.matched_keywords:
        with st.expander("🔑 Matched Keywords from Job Description"):
            keywords_html = ""
            for keyword in result.matched_keywords:
                keywords_html += f'<span class="skill-tag skill-matched">{keyword}</span>'
            st.markdown(f'<div>{keywords_html}</div>', unsafe_allow_html=True)


def create_download_button(result: AnalysisResult, resume_name: str):
    """Create download button for PDF report."""
    report_generator = ReportGenerator()
    pdf_bytes = report_generator.generate_report(result, resume_name)
    
    st.download_button(
        label="📥 Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"resume_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        help="Download a comprehensive PDF report of your resume analysis"
    )


def display_sidebar():
    """Display sidebar with information and instructions."""
    with st.sidebar:
        st.markdown("## 📋 How to Use")
        st.markdown("""
        1. **Upload Resume**: Upload your resume in PDF format
        2. **Paste Job Description**: Copy and paste the full job description
        3. **Analyze**: Click the analyze button to get your ATS score
        4. **Review Results**: Check matched/missing skills and recommendations
        5. **Download Report**: Get a detailed PDF report for your records
        """)
        
        st.markdown("---")
        
        st.markdown("## 📊 Score Breakdown")
        st.markdown("""
        - **Skill Match (40%)**: How many required skills you have
        - **Content Similarity (35%)**: Overall content alignment using TF-IDF
        - **Keyword Match (25%)**: Important keywords from job description
        """)
        
        st.markdown("---")
        
        st.markdown("## 🎯 Score Guide")
        st.markdown("""
        - **80-100**: Excellent match
        - **60-79**: Good match
        - **40-59**: Fair match
        - **0-39**: Needs improvement
        """)
        
        st.markdown("---")
        
        st.markdown("## ℹ️ About")
        st.markdown(f"""
        **{Config.APP_TITLE}** v{Config.VERSION}
        
        An AI-powered resume analysis tool that uses 
        NLP techniques to match your resume against 
        job descriptions, providing actionable insights 
        to improve your application success rate.
        """)


def main():
    """Main application entry point."""
    # Setup page configuration and styling
    setup_page()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'resume_name' not in st.session_state:
        st.session_state.resume_name = None
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=['pdf'],
            help="Upload your resume in PDF format for analysis"
        )
        
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    with col2:
        st.markdown("### 📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Paste the full job description here...",
            help="Copy and paste the complete job description from the job posting"
        )
    
    # Analyze button
    st.markdown("---")
    
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_button = st.button(
            "🔍 Analyze Resume",
            use_container_width=True,
            type="primary"
        )
    
    # Process analysis
    if analyze_button:
        if not uploaded_file:
            st.error("⚠️ Please upload a resume PDF file")
        elif not job_description or len(job_description.strip()) < 50:
            st.error("⚠️ Please provide a detailed job description (at least 50 characters)")
        else:
            try:
                with st.spinner("🔄 Analyzing your resume..."):
                    # Initialize processors
                    text_processor = TextProcessor()
                    ats_engine = ATSEngine()
                    
                    # Extract text from PDF
                    resume_text = text_processor.extract_text_from_pdf(uploaded_file)
                    
                    if not resume_text or len(resume_text.strip()) < 50:
                        st.error("⚠️ Could not extract sufficient text from the PDF. Please ensure your resume is text-based, not a scanned image.")
                    else:
                        # Perform analysis
                        result = ats_engine.analyze(resume_text, job_description)
                        
                        # Store results in session state
                        st.session_state.analysis_result = result
                        st.session_state.resume_name = uploaded_file.name
                        
                        st.success("✅ Analysis complete!")
                        
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                logging.error(f"Analysis error: {e}")
    
    # Display results if available
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Score section
        display_score_section(result)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Skills section
        display_skills_section(result)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Keywords section
        display_keywords_section(result)
        
        # Insights section
        display_insights_section(result)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Download report button
        st.markdown("### 📥 Download Report")
        create_download_button(result, st.session_state.resume_name)
        
        # Summary statistics
        with st.expander("📈 Detailed Statistics"):
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Total Matched Skills", len(result.matched_skills))
            with stat_col2:
                st.metric("Missing Skills", len(result.missing_skills))
            with stat_col3:
                st.metric("Additional Skills", len(result.resume_only_skills))
            with stat_col4:
                st.metric("Matched Keywords", len(result.matched_keywords))


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    main()
