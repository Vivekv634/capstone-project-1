import os
import re
import logging
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from tqdm import tqdm
from PIL import Image
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def post_process_ocr_text(text: str) -> str:
    """Post-process OCR text to correct common errors."""
    # Common OCR substitutions
    corrections = {
        r'\bl\b': 'I',  # l -> I (common in names)
        r'\b1\b': 'I',  # 1 -> I
        r'\b0\b': 'O',  # 0 -> O
        r'\b8\b': 'B',  # 8 -> B
        r'\b\|\b': 'I',  # | -> I
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    # Additional cleaning: remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def detect_language(text: str) -> str:
    """Detect language of text."""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English

def extract_text_from_pdf(pdf_path):
    """Extract text from both text-based and scanned PDFs with fallback."""
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
        doc.close()

        # If text-based extraction failed (e.g., scanned resume)
        if len(text.strip()) < 100:
            logging.info(f"Text extraction low for {pdf_path}, falling back to OCR.")
            text = ocr_pdf(pdf_path)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""  # Return empty on failure

def ocr_pdf(pdf_path):
    """Perform OCR on scanned PDF pages with detected language and post-processing."""
    images = convert_from_path(pdf_path, dpi=300)
    text_parts = []
    for img in images:
        # Preprocess image for better OCR
        img = img.convert('L')  # Grayscale
        page_text = pytesseract.image_to_string(img, lang='eng')  # Start with English
        text_parts.append(post_process_ocr_text(page_text))
    full_text = ' '.join(text_parts)
    # Detect language and re-OCR if not English
    lang = detect_language(full_text)
    if lang != 'en' and lang in pytesseract.get_languages():
        text_parts = []
        for img in images:
            img = img.convert('L')
            page_text = pytesseract.image_to_string(img, lang=lang)
            text_parts.append(post_process_ocr_text(page_text))
        full_text = ' '.join(text_parts)
    return full_text

def process_single_pdf(args):
    """Process a single PDF for parallel execution."""
    filename, input_dir, output_dir = args
    try:
        pdf_path = os.path.join(input_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        out_file = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        return f"Processed {filename}"
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return f"Failed {filename}"

def process_all_pdfs(input_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    """Batch process all resume PDFs with parallelization."""
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning("No PDF files found in input directory.")
        return

    args_list = [(filename, input_dir, output_dir) for filename in pdf_files]
    with ThreadPoolExecutor(max_workers=min(4, len(pdf_files))) as executor:
        results = list(tqdm(executor.map(process_single_pdf, args_list), total=len(pdf_files)))
    for result in results:
        if "Failed" in result:
            logging.error(result)
        else:
            logging.info(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_all_pdfs()
    print("✅ OCR extraction completed. Check 'data/extracted/' for results.")