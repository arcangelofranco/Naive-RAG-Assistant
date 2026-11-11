FROM python:3.12-slim

WORKDIR /app

# deps + model weights cached independently of app code
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py && pip install --no-cache-dir .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
COPY knowledge ./knowledge

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
