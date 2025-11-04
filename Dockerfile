# Multi-stage build to reduce image size
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps with caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"

# Download spaCy model
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Copy source code (for any build-time needs, if applicable)
COPY src/ ./src/
COPY pipeline.py .

# Runtime stage
FROM python:3.11-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy NLTK data
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy spaCy model
COPY --from=builder /usr/local/lib/python3.11/site-packages/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm

# Set working directory
WORKDIR /app

# Copy source and data
COPY src/ ./src/
COPY data/ ./data/
COPY pipeline.py .

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1
