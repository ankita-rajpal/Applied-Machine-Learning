# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py score.py ./  
COPY models/ ./models/  

COPY templates/ ./templates/

# Expose port used by Flask
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
