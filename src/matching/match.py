import os
import json
import re
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Optional
# from collections import Counter
import spacy
import nltk
# from nltk.corpus import wordnet, stopwords
import torch
from datetime import datetime

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Ensure NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuration for matching weights and paths
MATCH_CONFIG = {
    "weights": {
        "keyword_overlap": 0.25,
        "skill_overlap": 0.35,
        "experience_score": 0.1,
        "text_similarity": 0.1,
        "embedding_similarity": 0.2
    },
    "experience_max_diff": 5  # For normalization
}

# Global cache for TF-IDF vectorizer to avoid re-initialization
tfidf_cache = {}

# Load transformer model for text generation
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
try:
    from transformers import pipeline
    generator = pipeline('text2text-generation', model='t5-small', device=device)
except Exception as e:
    print(f"Warning: Could not load transformer model: {e}")
    generator = None

# Load sentence transformer model for embeddings
try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded embedding model on {'GPU' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"Warning: Could not load embedding model: {e}")
    embed_model = None

def generate_reasoning(details: Dict[str, float]) -> str:
    """Generate natural language reasoning using transformer with improved prompt."""
    if not generator:
        # Dynamic reasoning based on configurable thresholds
        reasoning_config = [
            ('keyword_overlap', 70, 50, "Excellent keyword alignment indicates strong relevance.", "Good keyword match with room for improvement."),
            ('skill_overlap', 70, 50, "Outstanding skills compatibility.", "Solid skills match."),
            ('experience_score', 80, 60, "Experience levels align perfectly.", "Experience is reasonably matched."),
            ('text_similarity', 70, 50, "High overall text similarity suggests strong fit.", "Moderate text similarity."),
            ('embedding_similarity', 70, 50, "Semantic embedding similarity shows deep contextual match.", "Good semantic alignment in embeddings.")
        ]
        reasoning_parts = []
        for metric, high_thresh, med_thresh, high_msg, med_msg in reasoning_config:
            if details[metric] > high_thresh:
                reasoning_parts.append(high_msg)
            elif details[metric] > med_thresh:
                reasoning_parts.append(med_msg)
        if not reasoning_parts:
            reasoning_parts.append("Limited alignment; consider skill gaps or experience mismatch.")
        return " ".join(reasoning_parts)

    # Improved prompt for better reasoning
    prompt = f"Explain the match quality: Keyword overlap {details['keyword_overlap']}%, skill overlap {details['skill_overlap']}%, experience score {details['experience_score']}%, text similarity {details['text_similarity']}%, embedding similarity {details['embedding_similarity']}%. Provide concise, professional reasoning."
    try:
        output = generator(prompt, max_length=60, num_return_sequences=1, temperature=0.7)
        generated = output[0]['generated_text'].strip()
        # Clean up
        if generated.startswith(prompt.split(':')[0]):
            generated = generated[len(prompt.split(':')[0]):].strip()
        return generated or "Generated reasoning unavailable."
    except Exception as e:
        print(f"Generation failed: {e}")
        return "Reasoning generation failed; using enhanced fallback."

def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_keywords(data: Dict[str, Any], is_jd: bool) -> List[str]:
    """Extract keywords from JD or resume data."""
    if is_jd:
        return data.get('raw_keywords', [])
    else:
        return data.get('keywords', [])

def extract_skills(data: Dict[str, Any], is_jd: bool) -> List[str]:
    """Extract skills from JD or resume data."""
    if is_jd:
        skills = data.get('skills', {})
        explicit = skills.get('explicit', [])
        implicit = skills.get('implicit', [])
        return explicit + implicit
    else:
        skills = data.get('skills', [])
        structured_skills = data.get('structured_info', {}).get('skills_details', [])
        return skills + structured_skills

def calculate_overlap(list1: List[str], list2: List[str]) -> float:
    """Calculate Jaccard similarity between two lists."""
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better embedding."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(' '.join(chunk))
    return chunks

def calculate_embedding_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using sentence transformers with chunking."""
    if not embed_model:
        return 0.0
    try:
        # Chunk texts
        chunks1 = chunk_text(text1)
        chunks2 = chunk_text(text2)
        if not chunks1 or not chunks2:
            # Fallback to whole text
            embeddings = embed_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        # Embed chunks and average
        emb1 = embed_model.encode(chunks1).mean(axis=0)
        emb2 = embed_model.encode(chunks2).mean(axis=0)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Embedding similarity failed: {e}")
        return 0.0

def extract_experience_level(data: Dict[str, Any], is_jd: bool) -> int:
    """Extract experience level (years)."""
    if is_jd:
        exp = data.get('experience_required', {})
        return exp.get('min_years', 0)
    else:
        # Improved: Parse dates from experience details to calculate total years
        exp_details = data.get('structured_info', {}).get('experience_details', [])
        total_years = 0
        current_year = datetime.now().year
        for exp in exp_details:
            period = exp.get('period', '')
            # Extract years from period like "2020-2023" or "Jan 2020 - Present"
            years = re.findall(r'\b(20\d{2})\b', period)
            if len(years) >= 2:
                start, end = int(years[0]), int(years[-1])
                if 'present' in period.lower() or end > current_year:
                    end = current_year
                total_years += max(0, end - start)
            elif len(years) == 1:
                # Assume 1 year if only one year mentioned
                total_years += 1
        return max(1, total_years) if total_years > 0 else len(exp_details)  # Fallback to count

def extract_text_from_data(data: Dict[str, Any], is_jd: bool) -> str:
    """Extract relevant text from JD or resume data."""
    if is_jd:
        text_parts = [
            data.get('job_title', ''),
            data.get('company_name', ''),
            str(data.get('experience_required', {})),
            str(data.get('skills', {})),
            ' '.join(data.get('responsibilities', [])),
            ' '.join(data.get('preferred_qualifications', []))
        ]
    else:
        sections = data.get('sections', {})
        text_parts = [
            data.get('name', ''),
            sections.get('education', ''),
            sections.get('experience', ''),
            sections.get('skills', ''),
            sections.get('projects', '')
        ]
    return ' '.join(text_parts).lower()

def match_jd_resume(jd_data: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced matching using keywords, skills, experience, and text similarity."""
    # Extract components
    jd_keywords = extract_keywords(jd_data, is_jd=True)
    resume_keywords = extract_keywords(resume_data, is_jd=False)
    jd_skills = extract_skills(jd_data, is_jd=True)
    resume_skills = extract_skills(resume_data, is_jd=False)
    jd_exp = extract_experience_level(jd_data, is_jd=True)
    resume_exp = extract_experience_level(resume_data, is_jd=False)

    # Calculate overlaps
    keyword_overlap = calculate_overlap(jd_keywords, resume_keywords)
    skill_overlap = calculate_overlap(jd_skills, resume_skills)

    # Experience match: closer levels are better
    exp_diff = abs(jd_exp - resume_exp)
    exp_score = max(0, 1 - exp_diff / MATCH_CONFIG["experience_max_diff"])  # Normalize using config

    # Text similarity with caching
    jd_text = extract_text_from_data(jd_data, is_jd=True)
    resume_text = extract_text_from_data(resume_data, is_jd=False)
    texts = [jd_text, resume_text]
    cache_key = hash((jd_text[:100], resume_text[:100]))  # Simple cache key
    if cache_key not in tfidf_cache:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_cache[cache_key] = vectorizer.fit(texts)
    else:
        vectorizer = tfidf_cache[cache_key]
    tfidf_matrix = vectorizer.transform(texts)
    text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Embedding similarity
    embedding_similarity = calculate_embedding_similarity(jd_text, resume_text)

    # Weighted score using config
    weights = MATCH_CONFIG["weights"]
    total_score = (keyword_overlap * weights["keyword_overlap"] +
                    skill_overlap * weights["skill_overlap"] +
                    exp_score * weights["experience_score"] +
                    text_similarity * weights["text_similarity"] +
                    embedding_similarity * weights["embedding_similarity"]) * 100
    score = round(total_score, 2)

    # Generate reasoning using transformer
    details_dict = {
        "keyword_overlap": round(keyword_overlap * 100, 2),
        "skill_overlap": round(skill_overlap * 100, 2),
        "experience_score": round(exp_score * 100, 2),
        "text_similarity": round(text_similarity * 100, 2),
        "embedding_similarity": round(embedding_similarity * 100, 2)
    }
    reasoning = generate_reasoning(details_dict)

    return {
        "score": score,
        "reasoning": reasoning,
        "details": details_dict
    }

def validate_data(data: Dict[str, Any], data_type: str) -> None:
    """Validate required keys in JD or resume data."""
    required_keys = {
        "jd": ["raw_keywords", "skills", "experience_required"],
        "resume": ["keywords", "skills", "structured_info"]
    }
    for key in required_keys[data_type]:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {data_type} data")

def main(jd_file: Optional[str] = None, resume_file: Optional[str] = None, output_file: Optional[str] = None):
    parser = argparse.ArgumentParser(description="Match JD and Resume")
    parser.add_argument("--jd", type=str, help="Path to JD JSON file")
    parser.add_argument("--resume", type=str, help="Path to Resume JSON file")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    # Use args or defaults
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    jd_file = args.jd or os.path.join(base_dir, "data", "processed", "jd1_jd.json")
    resume_file = args.resume or os.path.join(base_dir, "data", "processed", "vivek_resume_tagged.json")
    output_file = args.output or os.path.join(base_dir, "data", "results", "match_result.json")

    if not os.path.exists(jd_file):
        raise FileNotFoundError(f"JD file not found: {jd_file}")
    if not os.path.exists(resume_file):
        raise FileNotFoundError(f"Resume file not found: {resume_file}")

    jd_data = load_json(jd_file)
    resume_data = load_json(resume_file)

    # Validate data
    validate_data(jd_data, "jd")
    validate_data(resume_data, "resume")

    match_result = match_jd_resume(jd_data, resume_data)

    output = {
        "jd_file": jd_file,
        "resume_file": resume_file,
        "match": match_result
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    # Save structured data for ML training
    training_data_file = os.path.join(os.path.dirname(output_file), "training_data.json")
    training_entry = {
        "features": {
            "keyword_overlap": match_result["details"]["keyword_overlap"] / 100,  # Normalize to 0-1
            "skill_overlap": match_result["details"]["skill_overlap"] / 100,
            "experience_score": match_result["details"]["experience_score"] / 100,
            "text_similarity": match_result["details"]["text_similarity"] / 100,
            "embedding_similarity": match_result["details"]["embedding_similarity"] / 100
        },
        "score": match_result["score"] / 100,  # Normalize score to 0-1
        "timestamp": datetime.now().isoformat()
    }

    # Load existing or create new
    if os.path.exists(training_data_file):
        with open(training_data_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    else:
        training_data = []

    training_data.append(training_entry)

    with open(training_data_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Matching complete — result saved in {output_file}, training data updated in {training_data_file}")

if __name__ == "__main__":
    main()