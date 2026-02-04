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
COPY visualize.py .
COPY feature_importance.py .
COPY model_comparison.py .

# Create directories for outputs
RUN mkdir -p models plots

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Note: Dataset (creditcard.csv) should be mounted as a volume when running
# Example: docker run -v $(pwd)/creditcard.csv:/app/creditcard.csv fraud-detection

# Default command - just validate the environment
CMD ["python", "-c", "print('Fraud Detection Docker Image Ready! Mount creditcard.csv to train.')"]
