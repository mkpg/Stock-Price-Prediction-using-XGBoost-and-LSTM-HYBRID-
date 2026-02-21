# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies (using the CPU-optimized version we set up)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Hugging Face uses 7860, so we use that as default)
EXPOSE 7860

# Start the application with Gunicorn
# Use 1 worker for low memory, or increase if your local machine has 8GB+ RAM
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "--preload"]
