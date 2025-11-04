import os
import re
import json
import logging
from tqdm import tqdm
import spacy
from typing import Dict, List
from multiprocessing import Pool, cpu_count

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using spaCy."""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = []
        entities[label].append(ent.text)
    return entities

def extract_experience_details(text: str) -> List[Dict[str, str]]:
    """Extract job details from experience section using NLP and regex."""
    doc = nlp(text)
    experiences = []
    sentences = [sent.text for sent in doc.sents]
    for sent in sentences:
        sent_doc = nlp(sent)
        orgs = [ent.text for ent in sent_doc.ents if ent.label_ == 'ORG']
        dates = [ent.text for ent in sent_doc.ents if ent.label_ == 'DATE']
        # Regex for periods like "2020-2023" or "Jan 2020 - Present"
        period_match = re.search(r'(\d{4}(?:\s*-\s*\d{4}|(?:\s*-\s*Present)))|((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*-\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}))', sent, re.IGNORECASE)
        period = period_match.group(0) if period_match else (dates[0] if dates else '')
        # Heuristic: first ORG as company
        if orgs:
            experiences.append({
                'company': orgs[0],
                'period': period,
                'description': sent
            })
    return experiences

def extract_education_details(text: str) -> List[Dict[str, str]]:
    """Extract education details using NLP."""
    doc = nlp(text)
    educations = []
    sentences = [sent.text for sent in doc.sents]
    for sent in sentences:
        sent_doc = nlp(sent)
        orgs = [ent.text for ent in sent_doc.ents if ent.label_ == 'ORG']
        degrees = [token.text for token in sent_doc if 'degree' in token.lemma_ or token.text.lower() in ['bachelor', 'master', 'phd', 'b.tech', 'm.tech']]
        if orgs or degrees:
            educations.append({
                'institution': orgs[0] if orgs else '',
                'degree': degrees[0] if degrees else '',
                'description': sent
            })
    return educations

def extract_skills_from_section(text: str) -> List[str]:
    """Extract skills from skills section using NLP, including verbs."""
    doc = nlp(text.lower())
    skills = []
    # Extract nouns, proper nouns, and verbs (e.g., "developed Python")
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and len(token.lemma_) > 3 and token.lemma_ not in ['skill', 'experience', 'developed', 'used', 'worked']:
            skills.append(token.lemma_)
    # Also check for skill-like phrases
    skill_phrases = re.findall(r'\b(?:proficient|experienced|knowledge)\s+in\s+([a-zA-Z\s]+)', text, re.IGNORECASE)
    for phrase in skill_phrases:
        skills.extend(phrase.split())
    return list(set(skills))[:25]  # Increased limit

SECTION_PATTERNS = {
    "contact": r"\b(contact|personal\s+information|phone|email|address)\b",
    "summary": r"\b(summary|profile|objective|about\s+me|professional\s+summary)\b",
    "experience": r"\b(experience|work\s+experience|employment|professional\s+experience|work\s+history)\b",
    "education": r"\b(education|academic|qualification|degree)\b",
    "skills": r"\b(skills|technical\s+skills|core\s+competencies|expertise)\b",
    "projects": r"\b(projects|personal\s+projects|portfolio)\b",
    "certifications": r"\b(certification|certificate|license)\b",
    "awards": r"\b(awards|achievement|honor|recognition)\b",
    "publications": r"\b(publication|paper|research)\b",
    "languages": r"\b(language|linguistic)\b",
    "interests": r"\b(interest|hobby|hobbies|activities)\b",
    "references": r"\b(reference)\b"
}

def split_sections(text):
    sections = {}
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    current_section = "summary"  # Default to summary for top content
    sections[current_section] = []

    for line in lines:
        matched = False
        for name, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, line.lower()):
                current_section = name
                if current_section not in sections:
                    sections[current_section] = []
                matched = True
                break
        sections[current_section].append(line)

    # Clean up sections: remove empty, join content
    cleaned_sections = {}
    for k, v in sections.items():
        content = " ".join(v).strip()
        if content:
            cleaned_sections[k] = content

    return cleaned_sections

def process_single_section_tagging(args):
    """Process a single file for parallel execution."""
    filename, input_dir, output_dir = args
    try:
        path = os.path.join(input_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load raw text for accurate position-based extraction
        raw_filename = filename.replace("_clean.json", ".txt")
        raw_path = os.path.join(os.path.dirname(input_dir), "extracted", raw_filename)
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        lines = raw_text.split('\n')
        sections_found = data.get("sections_found", {})

        if sections_found:
            # Use positions for accurate extraction
            sorted_sections = sorted(sections_found.items(), key=lambda x: x[1])
            parsed_sections = {}
            prev_end = 0
            for i, (section_name, start_line) in enumerate(sorted_sections):
                # Add summary for content before first section
                if i == 0 and start_line > 0:
                    summary_content = '\n'.join(lines[0:start_line]).strip()
                    if summary_content:
                        parsed_sections["summary"] = summary_content
                end_line = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(lines)
                section_content = '\n'.join(lines[start_line:end_line]).strip()
                if section_content:
                    parsed_sections[section_name] = section_content
                prev_end = end_line
            # Add remaining content to summary if no other section
            if prev_end < len(lines):
                remaining = '\n'.join(lines[prev_end:]).strip()
                if remaining:
                    if "summary" in parsed_sections:
                        parsed_sections["summary"] += "\n" + remaining
                    else:
                        parsed_sections["summary"] = remaining
        else:
            # Fallback to pattern-based splitting on cleaned text
            text = data["cleaned_text"]
            parsed_sections = split_sections(text)

        data["sections"] = parsed_sections

        # Extract structured information using NLP
        structured_info = {}
        if "experience" in parsed_sections:
            exp_details = extract_experience_details(parsed_sections["experience"])
            # Validate: ensure some have periods
            if exp_details and any(d['period'] for d in exp_details):
                structured_info["experience_details"] = exp_details
            else:
                logging.warning(f"No valid periods in experience for {filename}")
        if "education" in parsed_sections:
            structured_info["education_details"] = extract_education_details(parsed_sections["education"])
        if "skills" in parsed_sections:
            skills = extract_skills_from_section(parsed_sections["skills"])
            if skills:
                structured_info["skills_details"] = skills
            else:
                logging.warning(f"No skills extracted for {filename}")
        # Extract entities from entire text
        entities = extract_entities(raw_text)
        structured_info["entities"] = entities

        data["structured_info"] = structured_info

        out_file = os.path.join(output_dir, filename.replace("_clean", "_tagged"))
        with open(out_file, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        return f"Processed {filename}"
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return f"Failed {filename}"

def tag_sections_in_all():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_clean.json")]
    if not files:
        logging.warning("No _clean.json files found in input directory.")
        return

    args_list = [(filename, INPUT_DIR, OUTPUT_DIR) for filename in files]
    with Pool(processes=min(cpu_count(), len(files))) as pool:
        results = list(pool.imap(process_single_section_tagging, args_list))
    for result in results:
        if "Failed" in result:
            logging.error(result)
        else:
            logging.info(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    tag_sections_in_all()
    print("✅ Section tagging complete — structured JSONs saved in 'data/processed/'")
