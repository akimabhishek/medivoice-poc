# Use an official lightweight Python image
FROM python:3.11-slim

# Hugging Face requires apps to run as a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application code into the container
COPY --chown=user . .

# Create necessary directories for runtime data and give permissions
RUN mkdir -p /app/app/data/policies /app/app/data/chroma /app/app/static

# Hugging Face Spaces ONLY open port 7860
EXPOSE 7860

# Command to run the application using Uvicorn on Port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]