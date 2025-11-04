import os
import re
import json
import logging
from tqdm import tqdm
from typing import Dict, List
import spacy
import nltk
from nltk.corpus import stopwords
from collections import Counter
from multiprocessing import Pool, cpu_count

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Ensure NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "extracted")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expanded common section headers (case-insensitive patterns)
SECTION_PATTERNS = {
    'contact': r'\b(contact|personal\s+information|phone|email|address|contact\s+details)\b',
    'summary': r'\b(summary|profile|objective|about\s+me|professional\s+summary|career\s+summary)\b',
    'experience': r'\b(experience|work\s+experience|employment|professional\s+experience|work\s+history|employment\s+history|job\s+experience)\b',
    'education': r'\b(education|academic|qualification|degree|educational\s+background)\b',
    'skills': r'\b(skills|technical\s+skills|core\s+competencies|expertise|competencies|abilities)\b',
    'projects': r'\b(projects|personal\s+projects|portfolio|project\s+experience)\b',
    'certifications': r'\b(certification|certificate|license|certifications|licenses)\b',
    'awards': r'\b(awards|achievement|honor|recognition|accolades)\b',
    'publications': r'\b(publication|paper|research|publications)\b',
    'languages': r'\b(language|linguistic|spoken\s+languages)\b',
    'interests': r'\b(interest|hobby|hobbies|activities|personal\s+interests)\b',
    'references': r'\b(reference|references)\b'
}

def identify_sections(text: str) -> Dict[str, int]:
    """Identify section headers and their positions in the text"""
    lines = text.split('\n')
    sections_found = {}

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if len(line_lower) < 3 or len(line_lower) > 50:
            continue

        for section_name, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, line_lower, re.IGNORECASE):
                # Check if line looks like a header (short, possibly uppercase/title case)
                if len(line.strip()) < 50 and (line.isupper() or line.istitle() or ':' in line):
                    if section_name not in sections_found:
                        sections_found[section_name] = i
                    break

    return sections_found

def extract_email(text: str) -> List[str]:
    """Extract email addresses"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def extract_phone(text: str) -> List[str]:
    """Extract phone numbers"""
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\+\d{10,15}'
    ]
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    return list(set(phones))

def extract_urls(text: str) -> List[str]:
    """Extract URLs (LinkedIn, GitHub, portfolio, etc.)"""
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    return re.findall(url_pattern, text)

def extract_name(text: str) -> str:
    """Extract name using spaCy NER for PERSON entities with improved fallback."""
    doc = nlp(text[:1500])  # Process first 1500 chars for better coverage
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    if persons:
        # Return the longest PERSON entity, assuming it's the full name
        return max(persons, key=len)
    # Enhanced fallback: Look for capitalized words at the start
    lines = text.split('\n')
    name_candidates = []
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if not line or re.search(r'@|https?://|\d{3}[-.\s]\d{3}', line):
            continue
        # Match full names: e.g., "John Doe" or "Jane Smith"
        match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line)
        if match:
            name_candidates.append(match.group(1))
    if name_candidates:
        return name_candidates[0]  # Return first match
    # Old method as last resort
    name_words = []
    current_word = ''
    for line in lines[:20]:
        line = line.strip()
        if not line:
            continue
        if len(line) <= 3 and line.isupper() and not line.endswith(':'):
            if current_word and len(current_word) >= 3 and len(line) == 1:
                name_words.append(current_word)
                current_word = line
            else:
                current_word += line
        else:
            match = re.match(r'^([A-Z]{1,3})', line)
            if match:
                if current_word:
                    current_word += match.group(1)
                else:
                    current_word = match.group(1)
            if re.search(r'@|https?://|\d{3}[-.\s]\d{3}', line):
                break
    if current_word:
        name_words.append(current_word)
    return ' '.join(name_words) if name_words else ""

def normalize_text(text: str) -> str:
    """Normalize abbreviations and common terms."""
    abbreviations = {
        r'\bml\b': 'machine learning',
        r'\bai\b': 'artificial intelligence',
        r'\bds\b': 'data science',
        r'\bdevops\b': 'development operations',
        r'\bci/cd\b': 'continuous integration/continuous deployment',
        r'\bapi\b': 'application programming interface',
        r'\bdb\b': 'database',
        r'\bui/ux\b': 'user interface/user experience'
    }
    for abbr, full in abbreviations.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    """Improved cleaning to handle OCR artifacts, preserve structure, and normalize."""
    # First, normalize abbreviations
    text = normalize_text(text)

    # Split into lines and handle broken words from OCR
    lines = text.split('\n')
    merged_words = []
    current_word = ''

    for line in lines:
        line = line.strip().replace('\n', ' ')  # Replace internal newlines with spaces
        # If line is a single uppercase letter or short cap word, merge into word
        if len(line) <= 3 and line.isupper() and not line.endswith(':'):
            current_word += line
        else:
            if current_word:
                merged_words.append(current_word)
                current_word = ''
            if line:
                merged_words.append(line)

    if current_word:
        merged_words.append(current_word)

    # Join words with spaces
    text = ' '.join(merged_words)

    # Replace multiple line breaks with one
    text = re.sub(r'\n+', '\n', text)

    # Normalize spacing within lines
    text = re.sub(r'[ \t]+', ' ', text)

    # Add newline before section headers (all-caps words)
    text = re.sub(r'(?<!\n)(?=[A-Z]{3,}[\s:])', r'\n', text)

    # Add newline before bullet points
    text = re.sub(r'(?<!\n)(?=[•\-–])', r'\n', text)

    # Remove extraneous symbols but keep common ones
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\+\n:/@]', '', text)

    # Trim spaces around newlines
    text = re.sub(r'\s*\n\s*', '\n', text)

    return text.strip()

def extract_keywords(text: str) -> List[str]:
    """Extract relevant keywords using NLP: lemmatize, remove stop words, count frequency."""
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token.lemma_) > 3]
    word_counts = Counter(lemmas)
    # Select top keywords by frequency, at least 2 occurrences
    top_keywords = [word for word, count in word_counts.most_common(50) if count > 1][:30]
    return top_keywords

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills using expanded predefined list and NLP."""
    skill_keywords = [
        # Programming languages
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab',
        # Web technologies
        'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express', 'django', 'flask', 'spring', 'asp.net',
        # Data and ML
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'machine learning', 'data science', 'artificial intelligence', 'deep learning', 'nlp', 'computer vision', 'opencv',
        # Cloud and DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'linux', 'windows', 'bash', 'powershell',
        # Other tools
        'api', 'rest', 'graphql', 'json', 'xml', 'excel', 'tableau', 'power bi', 'hadoop', 'spark', 'kafka'
    ]
    doc = nlp(text.lower())
    skills = [skill for skill in skill_keywords if skill in doc.text]
    # Also extract nouns that might be skills
    nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and len(token.lemma_) > 3 and token.lemma_ not in stop_words]
    unique_nouns = list(set(nouns))[:15]  # Increased limit
    skills.extend(unique_nouns)
    return list(set(skills))

def process_single_resume(args):
    """Process a single resume file for parallel execution."""
    filename, input_dir, output_dir = args
    try:
        path = os.path.join(input_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_text(text)

        # Extract contact information
        contact_info = {
            'emails': extract_email(text),
            'phones': extract_phone(text),
            'urls': extract_urls(text)
        }

        name = extract_name(cleaned)

        # Identify sections
        sections_found = identify_sections(text)

        # Extract keywords and skills from entire text
        keywords = extract_keywords(text)
        skills = extract_skills_from_text(text)

        out_file = os.path.join(output_dir, filename.replace(".txt", "_clean.json"))
        with open(out_file, "w", encoding="utf-8") as out:
            json.dump({
                "filename": filename,
                "cleaned_text": cleaned,
                "name": name,
                "contact_info": contact_info,
                "sections_found": sections_found,
                "keywords": keywords,
                "skills": skills
            }, out, indent=2)
        return f"Processed {filename}"
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return f"Failed {filename}"

def preprocess_all_resumes(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    if not txt_files:
        logging.warning("No TXT files found in input directory.")
        return

    args_list = [(filename, input_dir, output_dir) for filename in txt_files]
    with Pool(processes=min(cpu_count(), len(txt_files))) as pool:
        results = list(tqdm(pool.imap(process_single_resume, args_list), total=len(txt_files)))
    for result in results:
        if "Failed" in result:
            logging.error(result)
        else:
            logging.info(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    preprocess_all_resumes()
    print("✅ Cleaned text saved in 'data/processed/' with preserved section hints.")
