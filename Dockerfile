# Use NVIDIA CUDA base image if possible, otherwise standard python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV MPLCONFIGDIR=/app/cache
ENV HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     libgl1     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000
RUN useradd -m -u 1000 user

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory with right permissions
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Switch to the "user" user
USER user

# Set home to /home/user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Copy app code (owned by user)
COPY --chown=user . .

# Expose port (HuggingFace default is 7860)
EXPOSE 7860

# Start command
# We use uvicorn to run the FastAPI app on port 7860
CMD ["python", "api.py", "--port", "7860"]
