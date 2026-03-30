# Medical Imaging coursework Dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV MPLBACKEND=Agg

RUN pip install --no-cache-dir \
	numpy==2.4.4 \
	scipy==1.17.1 \
	scikit-image==0.26.0 \
	matplotlib==3.10.8 \
	PyWavelets==1.9.0 \
	pytest==9.0.2 \
	sphinx==9.1.0 \
	ipykernel==7.2.0

COPY pyproject.toml ./
COPY README.md ./
COPY med_im/ ./med_im/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY data/ ./data/

# Default: run the unit tests
CMD ["pytest", "tests/", "-v"]
