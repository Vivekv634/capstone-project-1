# AI ATS — Automated Resume Screening & Matching Pipeline

## 🧠 Introduction

**AI-ATS** is an automated Applicant Tracking System designed to process candidate resumes and job descriptions intelligently.
It extracts, cleans, and structures text using OCR and NLP techniques, then performs semantic matching between resumes and JDs to identify the best fit.

The system is fully containerized with Docker, supporting scalable deployments for recruitment automation workflows.

---

## 📚 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [Pipeline Overview](#-pipeline-overview)
6. [Configuration](#-configuration)
7. [Dependencies](#-dependencies)
8. [Docker Deployment](#-docker-deployment)
9. [Troubleshooting](#-troubleshooting)
10. [License](#-license)

---

## ✨ Features

* **OCR-Powered Resume Extraction:** Converts PDFs into machine-readable text using Tesseract and Poppler.
* **Automated Preprocessing:** Cleans and tokenizes resume text for uniformity.
* **Section Tagging:** Identifies key sections (Education, Experience, Skills, etc.) using rule-based or NLP-driven tagging.
* **Job Description Parsing:** Structures and preprocesses job postings for compatibility with resumes.
* **Intelligent Matching:** Compares processed resumes and JDs to output ranked match scores.
* **Logging & Validation:** Comprehensive logging and validation steps for pipeline integrity.
* **Containerized Environment:** Multi-stage Docker setup ensures reproducibility and efficiency.

---

## 📁 Project Structure

```
capstone-project-1/
│
├── pipeline.py              # Main orchestrator for end-to-end ATS workflow
├── Dockerfile               # Multi-stage Docker build configuration
├── requirements.txt         # Python dependencies
├── data/                    # Input/output directory (raw, extracted, processed)
├── src/                     # Core processing modules
│   ├── ocr/                 # Text extraction utilities
│   ├── preprocessing/       # Cleaning, parsing, and tagging modules
│   └── matching/            # Job-resume matching algorithms
└── logs/                    # Pipeline logs (auto-generated)
```

---

## ⚙️ Installation

### Prerequisites

* Python 3.11+
* Tesseract OCR and Poppler utilities
* Docker (optional but recommended)

### Manual Setup

```bash
git clone https://github.com/Vivekv634/capstone-project-1.git
cd ai-ats
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the entire pipeline:

```bash
python pipeline.py
```

The system will:

1. Extract text from PDF resumes and job descriptions.
2. Preprocess and clean all text data.
3. Tag and structure resume content.
4. Perform job-resume matching.
5. Save logs and results under `data/processed/`.

Example outputs:

```
data/processed/resume_tagged.json
data/processed/match_result.json
```

---

## 🧩 Pipeline Overview

1. **OCR Extraction** – Reads all resumes from `data/raw/` and converts to text.
2. **JD Processing** – Extracts structured data from job description files.
3. **Preprocessing** – Cleans, tokenizes, and normalizes text.
4. **Section Tagging** – Segments resumes into labeled sections.
5. **Matching** – Calculates similarity scores between resume and JD representations.
6. **Validation & Logging** – Ensures all outputs are generated and records progress.

---

## 🛠 Configuration

You can modify paths or logging behavior inside `CONFIG` in `pipeline.py`:

```python
CONFIG = {
    "data_dir": "data",
    "raw_dir": "data/raw",
    "jd_dir": "data/jd",
    "extracted_dir": "data/extracted",
    "processed_dir": "data/processed",
    "log_level": "INFO"
}
```

---

## 📦 Dependencies

From `requirements.txt` (not listed explicitly in your archive but inferred from code):

* `nltk`
* `spacy`
* `pdfminer.six`
* `pytesseract`
* `pandas`
* `numpy`
* `scikit-learn`
* `concurrent.futures` (built-in)
* `tqdm` (optional progress bar)

---

## 🐳 Docker Deployment

Build the image:

```bash
docker build -t ai-ats .
```

Run the container:

```bash
docker run -v $(pwd)/data:/app/data ai-ats
```

Health checks are included by default, ensuring the container is responsive.

---

## 🧯 Troubleshooting

* **Missing OCR dependencies**
  Ensure `tesseract-ocr` and `poppler-utils` are installed or available inside your container.

* **Missing models**
  If spaCy or NLTK models aren’t downloaded automatically, run:

  ```bash
  python -m spacy download en_core_web_sm
  python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
  ```

* **File not found errors**
  Check that your input resumes and job descriptions exist in `data/raw/` and `data/jd/` respectively.

---

## 🪪 License

This project is released under the **MIT License** — free for personal and commercial use with attribution.

---
