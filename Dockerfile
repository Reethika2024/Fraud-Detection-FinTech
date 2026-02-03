# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.py .
COPY preprocess.py .
COPY train_model.py .
COPY evaluate.py .
COPY main.py .

# Copy dataset (if available)
COPY creditcard.csv . 2>/dev/null || true

# Create directories for outputs
RUN mkdir -p models plots

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py", "--mode", "train"]