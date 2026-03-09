# Medical Imaging coursework Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Installing dependencies first
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Copying package and notebooks
COPY med_im/ ./med_im/
COPY solutions.ipynb ./

# Uncomment when you have a tests/ directory:
# COPY tests/ ./tests/

# Default: run tests (or override with e.g. jupyter notebook)
CMD ["pytest", "tests/", "-v"]
