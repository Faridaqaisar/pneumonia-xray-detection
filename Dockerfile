# Dockerfile for deploying the FastAPI backend to Hugging Face Spaces.
# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better build caching — only reinstalls if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (api/, src/, models/, etc.)
COPY . .

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]