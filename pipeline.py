import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.append('src')

from ocr.extract_text import process_all_pdfs
from preprocessing.clean_text import preprocess_all_resumes
from preprocessing.section_parser import tag_sections_in_all
from preprocessing.jd_preprocessing import process_all_jds
from matching.match import main

# Configuration (can be moved to env vars or config file)
CONFIG = {
    "data_dir": os.path.join(os.getcwd(), "data"),
    "raw_dir": os.path.join(os.getcwd(), "data", "raw"),
    "jd_dir": os.path.join(os.getcwd(), "data", "jd"),
    "extracted_dir": os.path.join(os.getcwd(), "data", "extracted"),
    "processed_dir": os.path.join(os.getcwd(), "data", "processed"),
    "log_level": "INFO"
}

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, CONFIG["log_level"]),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(CONFIG["data_dir"], "pipeline.log")),
            logging.StreamHandler()
        ]
    )

def validate_step(step_name, expected_files):
    """Validate that expected files exist after a step."""
    missing = []
    for file in expected_files:
        if not os.path.exists(file):
            missing.append(file)
    if missing:
        logging.error(f"Step '{step_name}' failed: Missing files {missing}")
        raise FileNotFoundError(f"Missing files after {step_name}: {missing}")
    logging.info(f"Step '{step_name}' validation passed.")

def run_pipeline():
    setup_logging()
    logging.info("Starting AI ATS Pipeline...")
    start_time = time.time()

    try:
        # Step 1 and 4 can run in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(process_all_pdfs): "Extracting text from PDFs",
                executor.submit(process_all_jds): "Processing job descriptions"
            }
            for future in as_completed(futures):
                step_name = futures[future]
                try:
                    future.result()
                    logging.info(f"Completed: {step_name}")
                except Exception as e:
                    logging.error(f"Error in {step_name}: {e}")
                    raise

        # Validate after parallel steps
        validate_step("Extraction and JD Processing", [
            os.path.join(CONFIG["extracted_dir"], "vivek_resume.txt"),  # Assuming sample file
            os.path.join(CONFIG["processed_dir"], "jd1_jd.json")
        ])

        # Step 2: Cleaning and preprocessing resumes
        logging.info("Step 2: Cleaning and preprocessing resumes...")
        step_start = time.time()
        preprocess_all_resumes()
        logging.info(f"Step 2 completed in {time.time() - step_start:.2f}s")

        # Validate
        validate_step("Resume Cleaning", [
            os.path.join(CONFIG["processed_dir"], "vivek_resume_clean.json")
        ])

        # Step 3: Parsing sections and structuring resumes
        logging.info("Step 3: Parsing sections and structuring resumes...")
        step_start = time.time()
        tag_sections_in_all()
        logging.info(f"Step 3 completed in {time.time() - step_start:.2f}s")

        # Validate
        validate_step("Section Parsing", [
            os.path.join(CONFIG["processed_dir"], "vivek_resume_tagged.json")
        ])

        # Step 5: Performing matching
        logging.info("Step 5: Performing matching...")
        step_start = time.time()
        main()
        logging.info(f"Step 5 completed in {time.time() - step_start:.2f}s")

        # Validate
        validate_step("Matching", [
            os.path.join(CONFIG["processed_dir"], "match_result.json")
        ])

        total_time = time.time() - start_time
        logging.info(f"Pipeline completed successfully in {total_time:.2f}s!")
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()