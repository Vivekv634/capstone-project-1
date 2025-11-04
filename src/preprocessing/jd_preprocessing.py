import os
import json
import re
import logging
import fitz  # PyMuPDF
from langdetect import detect
from typing import Dict, Any, List
import spacy
import nltk
from nltk.corpus import stopwords
from multiprocessing import Pool, cpu_count

# Expanded implicit skills mapping
IMPLICIT_SKILLS_MAP = {
    'python': ['data analysis', 'scripting', 'programming', 'automation'],
    'java': ['object-oriented programming', 'enterprise development'],
    'javascript': ['web development', 'frontend', 'dynamic scripting'],
    'c++': ['system programming', 'performance optimization'],
    'sql': ['database management', 'data querying', 'relational databases'],
    'tensorflow': ['machine learning', 'deep learning', 'neural networks', 'ai frameworks'],
    'pytorch': ['machine learning', 'deep learning', 'neural networks', 'research'],
    'scikit-learn': ['machine learning', 'data science', 'predictive modeling'],
    'react': ['frontend development', 'javascript', 'web development', 'ui components'],
    'node.js': ['backend development', 'javascript', 'server-side scripting'],
    'aws': ['cloud computing', 'infrastructure', 'scalability'],
    'docker': ['containerization', 'devops', 'microservices'],
    'kubernetes': ['containerization', 'orchestration', 'devops'],
    'git': ['version control', 'collaboration', 'code management'],
    'linux': ['system administration', 'devops', 'scripting'],
    'mysql': ['database management', 'sql', 'relational databases'],
    'postgresql': ['database management', 'sql', 'advanced querying'],
    'mongodb': ['database management', 'nosql', 'document databases'],
    'html': ['web development', 'frontend', 'markup'],
    'css': ['web development', 'styling', 'frontend'],
    'pandas': ['data analysis', 'data manipulation', 'data science'],
    'numpy': ['data analysis', 'scientific computing', 'mathematics'],
    'matplotlib': ['data visualization', 'data analysis', 'plotting'],
    'seaborn': ['data visualization', 'data analysis', 'statistics'],
    'jupyter': ['data science', 'interactive computing', 'prototyping'],
    'spark': ['big data', 'data processing', 'distributed computing'],
    'hadoop': ['big data', 'distributed computing', 'data storage'],
    'ml': ['machine learning', 'data science', 'predictive analytics'],
    'ai': ['artificial intelligence', 'automation', 'intelligent systems'],
    'api': ['web development', 'backend development', 'integration', 'rest'],
    'azure': ['cloud computing', 'microsoft ecosystem'],
    'gcp': ['cloud computing', 'google services'],
    'jenkins': ['ci/cd', 'automation', 'devops'],
    'ansible': ['automation', 'devops', 'configuration management'],
    'terraform': ['infrastructure as code', 'cloud provisioning'],
    'kafka': ['data streaming', 'event-driven architecture'],
    'elasticsearch': ['search', 'data indexing', 'analytics'],
    'redis': ['caching', 'data structures', 'performance'],
    'graphql': ['api design', 'query language'],
    'flutter': ['mobile development', 'cross-platform'],
    'swift': ['ios development', 'mobile'],
    'kotlin': ['android development', 'mobile'],
    'go': ['system programming', 'concurrency'],
    'rust': ['system programming', 'memory safety'],
    'ruby': ['web development', 'scripting'],
    'php': ['web development', 'server-side'],
    'scala': ['big data', 'functional programming'],
    'r': ['statistics', 'data analysis'],
    'tableau': ['data visualization', 'business intelligence'],
    'power bi': ['data visualization', 'business intelligence'],
    'excel': ['data analysis', 'spreadsheets'],
    'sap': ['enterprise software', 'business processes'],
    'oracle': ['database management', 'enterprise databases']
}

def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or TXT file."""
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_job_title(text: str) -> str:
    """Extract job title using regex patterns and spaCy fallback."""
    patterns = [
        r'job details[:\s]*([^\n]+)',
        r'job title[:\s]*([^\n]+)',
        r'position[:\s]*([^\n]+)',
        r'role[-\s]*([^\n]+)',
        r'we are looking for[:\s]*([^\n]+)',
        r'hiring[:\s]*([^\n]+)',
        r'opening for[:\s]*([^\n]+)',
        r'senior\s+([^\n]+)',
        r'junior\s+([^\n]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # spaCy fallback: Look for ORG or WORK_OF_ART entities
    doc = spacy.load('en_core_web_sm')(text[:1000])
    for ent in doc.ents:
        if ent.label_ in ['WORK_OF_ART', 'ORG'] and len(ent.text) > 3:
            return ent.text
    # Fallback: first meaningful line
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 5 and not re.search(r'\d|@|http', line):
            return line
    return None

def extract_company_name(text: str) -> str:
    """Extract company name using regex and spaCy fallback."""
    patterns = [
        r'company name[-\s]*([^\n]+)',
        r'company[:\s]*([^\n]+)',
        r'at ([^\n]+)',
        r'join ([^\n]+)',
        r'working at ([^\n]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:inc|ltd|llc|corp|corporation|company)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # spaCy fallback: Look for ORG entities
    doc = spacy.load('en_core_web_sm')(text[:1000])
    orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    if orgs:
        return orgs[0]  # Return first ORG
    return None

def extract_employment_type(text: str) -> str:
    """Extract employment type."""
    if re.search(r'full.?time', text, re.IGNORECASE):
        return "Full-time"
    elif re.search(r'part.?time', text, re.IGNORECASE):
        return "Part-time"
    elif re.search(r'intern', text, re.IGNORECASE):
        return "Internship"
    elif re.search(r'contract', text, re.IGNORECASE):
        return "Contract"
    return "Full-time"  # default

def extract_experience(text: str) -> Dict[str, Any]:
    """Extract experience requirements."""
    exp_match = re.search(r'(?:experience|years of experience)[:\s]*(\d+)(?:\s*-\s*(\d+))?\s*years?', text, re.IGNORECASE)
    if exp_match:
        min_years = int(exp_match.group(1))
        max_years = int(exp_match.group(2)) if exp_match.group(2) else min_years + 2
        level = "Senior" if min_years >= 5 else "Mid" if min_years >= 2 else "Entry"
    else:
        min_years = 0  # for freshers
        max_years = 0
        level = "Entry"
    return {"min_years": min_years, "max_years": max_years, "level": level}

def extract_skills(text: str) -> Dict[str, List[str]]:
    """Extract skills."""
    explicit = []
    # Find skills section
    skills_match = re.search(r'skills?[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
    if skills_match:
        skills_text = skills_match.group(1)
        explicit = [s.strip() for s in re.split(r'[,;•\-*]', skills_text) if s.strip() and len(s.strip()) > 2]
    else:
        # Extract from whole text with expanded list
        skill_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab',
            'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express', 'django', 'flask', 'spring', 'asp.net',
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'machine learning', 'data science', 'artificial intelligence', 'deep learning', 'nlp', 'computer vision', 'opencv',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'linux', 'windows', 'bash', 'powershell',
            'api', 'rest', 'graphql', 'json', 'xml', 'excel', 'tableau', 'power bi', 'hadoop', 'spark', 'kafka'
        ]
        explicit = [kw for kw in skill_keywords if kw in text.lower()]

    implicit = []
    for skill in explicit:
        skill_lower = skill.lower()
        if skill_lower in IMPLICIT_SKILLS_MAP:
            implicit.extend(IMPLICIT_SKILLS_MAP[skill_lower])

    implicit = list(set(implicit))  # unique

    soft_skills = []  # Not extracting for simplicity

    return {"explicit": explicit, "implicit": implicit, "soft_skills": soft_skills}

def extract_responsibilities(text: str) -> List[str]:
    """Extract responsibilities."""
    resp_match = re.search(r'responsibilities?[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
    if resp_match:
        resp_text = resp_match.group(1)
        resp = re.split(r'[•\-*]', resp_text)
        resp = [r.strip() for r in resp if r.strip() and len(r.strip()) > 10]
    else:
        # Extract sentences with action verbs
        sentences = re.split(r'[.!?]', text)
        resp = [s.strip() for s in sentences if re.search(r'\b(?:develop|build|manage|create|design|implement)\b', s, re.IGNORECASE)][:5]
    return resp

def extract_qualifications(text: str) -> List[str]:
    """Extract preferred qualifications."""
    qual = []
    # Extract degrees
    degrees_match = re.search(r'target degrees.*?([^\n]+)', text, re.IGNORECASE)
    if degrees_match:
        qual.append(f"Degrees: {degrees_match.group(1).strip()}")
    # Extract batch
    batch_match = re.search(r'batch[:\s]*([^\n]+)', text, re.IGNORECASE)
    if batch_match:
        qual.append(f"Batch: {batch_match.group(1).strip()}")
    # Extract CTC
    ctc_match = re.search(r'ctc[:\s]*([^\n]+)', text, re.IGNORECASE)
    if ctc_match:
        qual.append(f"CTC: {ctc_match.group(1).strip()}")
    # Extract cut-off
    cutoff_match = re.search(r'% cut-off(.*?)(?:\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
    if cutoff_match:
        cutoff_text = cutoff_match.group(1)
        cutoff_lines = [line.strip() for line in cutoff_text.split('\n') if line.strip() and not line.startswith('•')]
        qual.extend(cutoff_lines)
    # Extract service agreement
    service_match = re.search(r'service agreement[:\s]*([^\n]+)', text, re.IGNORECASE)
    if service_match:
        qual.append(f"Service Agreement: {service_match.group(1).strip()}")
    # Extract joining date
    joining_match = re.search(r'target joining date[:\s]*([^\n]+)', text, re.IGNORECASE)
    if joining_match:
        qual.append(f"Target Joining Date: {joining_match.group(1).strip()}")
    # Extract work shift
    shift_match = re.search(r'work shift.*?([^\n]+)', text, re.IGNORECASE)
    if shift_match:
        qual.append(f"Work Shift: {shift_match.group(1).strip()}")
    # Extract locations
    location_match = re.search(r'job location[:\s]*(.*?)(?:\n\n|$)', text, re.IGNORECASE | re.DOTALL)
    if location_match:
        locations = [loc.strip() for loc in location_match.group(1).split('\n') if loc.strip()]
        qual.append(f"Job Locations: {', '.join(locations)}")
    return qual

def extract_domain(text: str) -> str:
    """Extract domain based on keywords."""
    text_lower = text.lower()
    if 'artificial intelligence' in text_lower or 'machine learning' in text_lower:
        return "Artificial Intelligence"
    elif 'i.t' in text_lower or 'information technology' in text_lower:
        return "Information Technology"
    elif 'web' in text_lower or 'frontend' in text_lower or 'backend' in text_lower:
        return "Web Development"
    elif 'data' in text_lower or 'analytics' in text_lower:
        return "Data Science"
    elif 'mobile' in text_lower or 'ios' in text_lower or 'android' in text_lower:
        return "Mobile Development"
    return None

def extract_language(text: str) -> str:
    """Detect language."""
    try:
        return detect(text)
    except:
        return "en"  # default

def extract_raw_keywords(text: str) -> List[str]:
    """Extract raw keywords using NLTK stop words and frequency filtering."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out stop words, short words, and non-alphabetic
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3 and w.isalpha()]
    # Count frequency
    from collections import Counter
    word_counts = Counter(filtered_words)
    # Select top 20 by frequency, requiring at least 2 occurrences
    top_keywords = [word for word, count in word_counts.most_common(30) if count > 1][:20]
    if len(top_keywords) < 10:
        top_keywords = [word for word, count in word_counts.most_common(20)]
    return top_keywords

def calculate_ambiguity_score(structured: Dict[str, Any]) -> float:
    """Calculate ambiguity score based on filled fields."""
    fields = [
        structured.get('job_title'),
        structured.get('company_name'),
        structured.get('employment_type'),
        structured.get('experience_required', {}).get('min_years'),
        len(structured.get('skills', {}).get('explicit', [])),
        len(structured.get('responsibilities', [])),
        len(structured.get('preferred_qualifications', [])),
        structured.get('domain'),
        structured.get('language')
    ]
    filled = sum(1 for f in fields if f)
    return filled / len(fields)

def detect_errors(text: str, structured: Dict[str, Any]) -> List[str]:
    """Detect errors and flag low quality."""
    errors = []
    if len(text.strip()) < 50:
        errors.append("Too short")
    if not structured.get('job_title'):
        errors.append("No Job Title Detected")
    if not structured.get('company_name'):
        errors.append("No Company Name Detected")
    if not structured.get('skills', {}).get('explicit'):
        errors.append("No Skills Mentioned")
    if not structured.get('responsibilities'):
        errors.append("No Responsibilities Mentioned")
    if structured.get('ambiguity_score', 0) < 0.5:
        errors.append("Low Ambiguity Score - Incomplete Extraction")
    return errors

def process_jd(text: str) -> Dict[str, Any]:
    """
    Process job description text using rule-based extraction.

    Args:
        text (str): The raw job description text.

    Returns:
        Dict[str, Any]: Structured JSON output.
    """
    text = clean_text(text)

    job_title = extract_job_title(text)
    company_name = extract_company_name(text)
    employment_type = extract_employment_type(text)
    experience_required = extract_experience(text)
    skills = extract_skills(text)
    responsibilities = extract_responsibilities(text)
    preferred_qualifications = extract_qualifications(text)
    domain = extract_domain(text)
    language = extract_language(text)
    raw_keywords = extract_raw_keywords(text)

    structured = {
        "job_title": job_title,
        "company_name": company_name,
        "employment_type": employment_type,
        "experience_required": experience_required,
        "skills": skills,
        "responsibilities": responsibilities,
        "preferred_qualifications": preferred_qualifications,
        "domain": domain,
        "language": language,
        "raw_keywords": raw_keywords,
        "ambiguity_score": calculate_ambiguity_score(locals()),
        "errors_detected": detect_errors(text, locals()),
        "parsed_successfully": True
    }

    return structured

def process_single_jd(args):
    """Process a single JD file for parallel execution."""
    filename, input_dir, output_dir = args
    try:
        file_path = os.path.join(input_dir, filename)
        text = extract_text_from_file(file_path)
        structured = process_jd(text)
        out_filename = filename.replace('.pdf', '_jd.json').replace('.txt', '_jd.json')
        out_path = os.path.join(output_dir, out_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(structured, f, indent=2)
        return f"Processed {filename}"
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return f"Failed {filename}"

def process_all_jds(input_dir: str = None, output_dir: str = None):
    """Batch process all JD files (PDF/TXT) in input_dir with parallelization."""
    if input_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_dir = os.path.join(base_dir, "data", "jd")
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.pdf', '.txt'))]
    if not files:
        logging.warning("No JD files found in input directory.")
        return

    args_list = [(filename, input_dir, output_dir) for filename in files]
    with Pool(processes=min(cpu_count(), len(files))) as pool:
        results = list(pool.imap(process_single_jd, args_list))
    for result in results:
        if "Failed" in result:
            logging.error(result)
        else:
            logging.info(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Automatically process all JD files in data/jd/
    process_all_jds()
    print("✅ JD processing complete — structured JSONs saved in 'data/processed/'")