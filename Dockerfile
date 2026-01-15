FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
# We presume 'models' has been populated by running training locally or in CI
COPY src/ src/
COPY app/ app/
COPY models/ models/
# Copy raw data for /similar endpoint (Authentic news database)
COPY data/raw/Authentic-48K.csv data/raw/Authentic-48K.csv

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
