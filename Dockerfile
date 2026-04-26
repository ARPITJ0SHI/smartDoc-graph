FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directory for persistent data
RUN mkdir -p data/uploads data/faiss

# Expose port
EXPOSE 8000

# Default command (will be overridden by Render service commands)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
