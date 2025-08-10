# Dockerfile
FROM python:3.11-slim

# Install system packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY deploy/app/main.py .
COPY deploy/app/model_loader.py .
COPY configs/ configs/
COPY src/ src/
COPY models/ models/
COPY .env .env

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Run app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]