# Create Dockerfile
dockerfile_content = '''# Multi-stage Docker build for ML System Optimizer
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    libc6-dev \\
    libffi-dev \\
    libssl-dev \\
    curl \\
    wget \\
    procps \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models /app/logs

# Set permissions
RUN chmod +x scripts/*.sh || true

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:5000/ || exit 1

# Default command
CMD ["python", "-m", "src.api.app"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir jupyter notebook ipython

# Expose additional ports for development
EXPOSE 8888

# Production stage
FROM base as production

# Remove development tools and clean up
RUN apt-get autoremove -y && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "src.api.app:app"]
'''

with open('ml_system_optimizer/docker/Dockerfile', 'w') as f:
    f.write(dockerfile_content)

print("✅ Dockerfile created successfully!")