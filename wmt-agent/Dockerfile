# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY wmt_agent.py .
COPY wmt_api.py .
COPY .env .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=wmt_api.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "wmt_api.py"]