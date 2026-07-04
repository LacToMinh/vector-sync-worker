# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY scraper.py state_manager.py uploader.py main.py ./

# Persistent data volumes (mount these in production):
#   /app/articles   — scraped Markdown files
#   /app/chroma_db  — ChromaDB vector store (must persist between runs!)
#   /app/state.json — delta tracking

# Run once and exit (not a server)
# Pass GEMINI_API_KEY as environment variable:
#   docker run -e GEMINI_API_KEY=AIzaSy-... optibot
CMD ["python", "main.py"]
