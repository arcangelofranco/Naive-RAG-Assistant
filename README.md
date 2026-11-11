<div align="center">

# Naive RAG Assistant

A retrieval-augmented generation service built on a curated software engineering knowledge base. It is a FastAPI application that uses PostgreSQL with pgvector for vector storage. Answers are grounded in retrieved context and validated before they reach the user, and every claim carries an inline citation pointing at a specific passage. When validation fails, the system refuses to answer rather than guessing.

The LLM backend is interchangeable. Google Gemini works natively, and so does any OpenAI-compatible endpoint, including Groq, Ollama and vLLM. Embeddings run locally with `all-MiniLM-L6-v2`.

This is an implementation exercise, not a production service. Every stage of the pipeline is explicit and inspectable, from parsing to chunking, from embedding to vector search, through prompt construction, generation and verification.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-0.8-336791)
![FastAPI](https://img.shields.io/badge/fastapi-%23009688.svg?style=&logo=fastapi&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=&logo=docker&logoColor=white)
![Google Gemini](https://img.shields.io/badge/google%20gemini-%238E75B2.svg?style=&logo=google%20gemini&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Hexagonal-blueviolet)

</div>

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)

---
## Project Structure

```text
root/
├── README.md
├── LICENSE
├── pyproject.toml                      # dependencies, packaging, pytest markers
├── Dockerfile                          # api image: dependencies + model weights cached before the code
├── docker-compose.yaml                 # db (pgvector/pg16) + api services
├── .env / .env.example                 # provider credentials and runtime settings
│
├── database/
│   └── schema.sql                      # tables, pgvector extension, indexes (applied on first init)
│
├── knowledge/                          # the corpus: 35 Markdown documents with YAML frontmatter
│
└── app/
    ├── main.py                         # FastAPI app, lifespan, OperationalError -> 503 handler
    ├── composition.py                  # composition root: builds providers, repositories, services
    ├── deps.py                         # FastAPI dependency wiring, per-app provider caching
    │
    ├── config/
    │   └── config.py                   # Settings (pydantic-settings) + get_settings() with lru_cache
    │
    ├── api/
    │   ├── schemas.py                  # request and response models
    │   ├── routes_health.py            # GET  /health
    │   ├── routes_ask.py               # POST /ask
    │   └── routes_debug.py             # GET  /debug/retrieve (retrieval without the LLM)
    │
    ├── domain/                         # pure logic: no I/O, no framework, no provider imports
    │   ├── docs.py                     # ParsedDoc, Section, Chunk, RetrievedChunk + text helpers
    │   ├── prompt.py                   # build_prompt, count_passages, parse_citations
    │   ├── grounding.py                # similarity filtering and citation validation
    │   └── instructions.py             # the grounded retrieval system prompt
    │
    ├── ingestion/
    │   ├── parser.py                   # frontmatter + splitting on "## " sections
    │   ├── chunker.py                  # section -> overlapping token-budgeted windows
    │   ├── ingestor.py                 # parse -> chunk -> embed -> persist
    │   └── corpus.py                   # corpus scan + `python -m app.ingestion.corpus` entry point
    │
    ├── providers/
    │   ├── embedding/
    │   │   ├── base.py                 # EmbeddingProvider protocol
    │   │   ├── token_counter.py        # TokenCounter protocol (embed + count_tokens + max_tokens)
    │   │   └── embedding_minilm.py     # sentence-transformers implementation
    │   └── llm/
    │       ├── base.py                 # LLMProvider protocol
    │       ├── llm_gemini.py           # google-genai implementation
    │       ├── llm_openai.py           # OpenAI-compatible implementation (Groq, Ollama, ...)
    │       ├── llm_retry.py            # retry decorator: provider Retry-After, else linear backoff
    │       ├── llm_cache.py            # conditional in-memory response cache
    │       └── llm_exceptions.py       # provider-independent error taxonomy
    │
    ├── repositories/
    │   ├── base.py                     # ChunkSearcher / KnowledgeWriter / EmbeddingFiller protocols
    │   ├── postgres.py                 # the single PostgreSQL adapter
    │   └── queries.py                  # SQL kept out of the adapter logic
    │
    ├── services/
    │   └── query_service.py            # retrieve -> filter -> prompt -> generate -> validate
    │
    └── database/
        └── db.py                       # psycopg connection pool with pgvector registration
```

---
## Installation

### Prerequisites

* **Docker** with **Compose v2**. The whole application runs inside containers, so no local Python installation is required.

* **An API key** for at least one LLM provider:
  * a **[Google AI Studio](https://aistudio.google.com/apikey)** key to use Gemini;
  * or a **[Groq](https://console.groq.com/keys)** key to use the OpenAI-compatible path.

The embedding model runs locally inside the container and needs no credentials. It is downloaded once during the Docker image build and baked into one of its layers, so the container does not need network access to produce embeddings at runtime.

### Setup

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/naive-rag-assistant.git
cd naive-rag-assistant
```

#### 2. Configure the environment

Create the `.env` file from the provided template:

```bash
cp .env.example .env
```

Then set `LLM_API_KEY` in `.env`. The default configuration uses Gemini. To use the OpenAI-compatible path instead, see the [Configuration Reference](#configuration-reference).

> **Windows:** make sure the `.env` file uses **LF** line endings.

#### 3. Build the Docker image

```bash
docker compose build
```

The first build can take several minutes, because it installs PyTorch and downloads the `all-MiniLM-L6-v2` embedding model weights. Both live in Docker layers placed before the application code is copied in, so code changes do not invalidate them and subsequent builds finish in a few seconds. See [Docker](docs/DESIGN.md#docker).

---
## Usage

### 1. Start the database

```bash
docker compose up -d db
```

On first startup, PostgreSQL automatically runs [`database/schema.sql`](database/schema.sql) through the `docker-entrypoint-initdb.d` hook. The script creates the `vector` extension and the three tables the application needs.

> [!NOTE]
> The initialization script runs only when the data volume is empty. If you change the schema later, see [Limitations](docs/DESIGN.md#limitations) for the update procedure.

You can confirm that the tables were created correctly with:

```bash
docker compose exec db psql -U postgres -d naive_rag_assistant -c "\dt"
```

### 2. Ingest the knowledge base

```bash
docker compose run --rm api python -m app.ingestion.corpus
```

The command locates the `knowledge/**/*.md` files, parses their frontmatter, splits each document into sections and chunks, computes the embedding of each chunk locally, and stores documents, chunks and vectors in PostgreSQL.

When it finishes, the output looks like this:

```text
35 documents, 179 chunks ingested.
```

The operation is idempotent: documents are updated by their `id`, while their chunks are deleted and rewritten. You can re-run ingestion after editing the corpus to refresh the index without creating duplicate rows.

### 3. Start the API

```bash
docker compose up -d api
curl http://localhost:8000/health
```
```json
{"status":"ok","database":true,"chunks_indexed":true,"embedding_model_loaded":true}
```

FastAPI serves the interactive documentation at **http://localhost:8000/docs**.

### 4. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the difference between an object adapter and a class adapter?"}'
```

> [!NOTE]
> **PowerShell 5.1** may decode the response as `Latin-1` instead of `UTF-8`, which garbles accented characters. Round-trip the response through `UTF-8` explicitly:
>
> ```powershell
> $body = @{ question = "What is the Facade pattern?" } | ConvertTo-Json
> $resp = Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8000/ask `
>     -Method Post -ContentType "application/json" `
>     -Body ([Text.Encoding]::UTF8.GetBytes($body))
> ([Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray()) | ConvertFrom-Json).answer
> ```

### 5. Inspect retrieval without spending an LLM call

```bash
curl "http://localhost:8000/debug/retrieve?q=adapter&top_k=5"
```

The endpoint embeds the query, runs the vector search, and returns the most relevant chunks together with their similarity scores. Every similarity figure reported in this document was collected through it, at no cost in LLM calls.

### Switching provider

The active LLM provider is configured entirely through environment variables, so no code change is required. To use Groq instead of Gemini, update the API key, provider, model and base URL in `.env`, then recreate the `api` container:

```bash
docker compose up -d --force-recreate api
```

For a one-off test that leaves `.env` untouched, override the variables directly:

```bash
docker compose run --rm --service-ports \
  -e LLM_PROVIDER=openai \
  -e LLM_MODEL=qwen/qwen3.8-27b \
  -e LLM_BASE_URL=https://api.groq.com/openai/v1 \
  -e LLM_API_KEY=... \
  api
```

The variables passed with `-e` apply only to the container created by that command.

---
## API Reference

| Method | Path              | Purpose                                                                                                                                                              |
| :----- | :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health`         | Reports application status, database reachability, whether the index is populated, and whether the embedding model is available. Returns `503` if the database is unreachable. |
| `POST` | `/ask`            | Runs the full pipeline: retrieval, grounding, generation and answer validation.                                                                                      |
| `GET`  | `/debug/retrieve` | Runs retrieval only, with no LLM call. Useful for inspecting results and tuning `TOP_K` and `SIMILARITY_THRESHOLD`.                                                   |

### `POST /ask`

#### Request

| Field      | Type     | Constraints                  | Description                                             |
| :--------- | :------- | :--------------------------- | :------------------------------------------------------ |
| `question` | `string` | 1 to 1000 characters, required | The user question.                                    |
| `top_k`    | `int`    | 1 to 20, optional            | Overrides the configured `TOP_K` for this request only. |

#### Response

| Field      | Type     | Description                                                                                                                                    |
| :--------- | :------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| `answer`   | `string` | The grounded answer, with inline `[n]` citations, or the expected refusal string.                                                              |
| `sources`  | `array`  | The passages supporting the answer. It is empty when the answer is refused.                                                                    |
| `grounded` | `bool`   | Whether the answer is supported by retrieved context. It is `false` when nothing clears the similarity threshold, or when citation validation fails. |

Each entry in `sources` contains `document_id`, `title`, `section`, `chunk_index` and `similarity`.

### Response states

| Situation                                                  |  HTTP | Body                                             |
| :--------------------------------------------------------- | :---: | :----------------------------------------------- |
| Request processed successfully                             | `200` | `grounded: true`, with the supporting `sources`. |
| Out-of-domain question, or invalid and fabricated citations | `200` | `grounded: false`, `sources: []`.               |
| Empty question, or longer than 1000 characters             | `422` | Pydantic validation error.                       |
| Provider rate limit, after 3 attempts                      | `429` | Response with `Retry-After` if the provider supplied one. |
| Provider unreachable, or the request timed out             | `503` | Neutral message.                                 |
| Database unreachable                                       | `503` | `"Database not accessible."`                     |
| Missing or invalid key, unknown model, or request rejected as malformed | `500` | FastAPI's generic error. Not retried, and it names nothing about the provider. |

> [!NOTE]
> No response ever names the LLM provider. The `429` and `503` handlers return an empty `detail` field, which prevents the configured backend from leaking to the client. This behavior is not currently covered by automated tests, as described in [Limitations](docs/DESIGN.md#limitations).

---
## Configuration Reference

All settings are defined by the `Settings` model in [`app/config/config.py`](app/config/config.py), built on `pydantic-settings`. Values are loaded from the `.env` file and can be overridden by real environment variables. `get_settings()` is decorated with `lru_cache`, so configuration is loaded once per process.

| Variable               |                            Default                            | Description                                                                                                                                                                     |
| :--------------------- | :-----------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLM_PROVIDER`         |                            `google`                           | Selects the LLM backend. `google` uses the Gemini SDK, while `openai` uses the OpenAI SDK against any compatible endpoint. Any other value is rejected at startup.               |
| `LLM_API_KEY`          |                           *(empty)*                           | The provider credential. It is required for both backends, and when it is missing an explicit error identifies the missing field.                                                |
| `LLM_MODEL`            |                           *(empty)*                           | The model identifier to use, for example `gemini-3.5-flash` or `qwen/qwen3.8-27b`.                                                                                              |
| `LLM_BASE_URL`         |                           *(empty)*                           | The LLM endpoint URL. It is ignored with `google` and required with `openai`, for example `https://api.groq.com/openai/v1`.                                                      |
| `EMBEDDING_MODEL`      |                           *(empty)*                           | The `sentence-transformers` model name used for embeddings. It must produce 384-dimensional vectors, so that it stays compatible with the database schema.                       |
| `EMBEDDINGS_TABLE`     |                       `chunk_embeddings`                      | The PostgreSQL table where vectors are stored. The value is injected as a quoted SQL identifier, which makes it possible to keep an alternative embedding space in a separate table. |
| `DATABASE_URL`         | `postgresql://postgres:postgres@localhost:5432/naive_rag_assistant` | The PostgreSQL connection string. Under Docker Compose the host is overridden with `db`, so that the API can reach the database container.                                       |
| `SIMILARITY_THRESHOLD` |                             `0.5`                             | The minimum cosine similarity for a chunk to enter the prompt. See [Retrieval and the Similarity Threshold](docs/DESIGN.md#retrieval-and-the-similarity-threshold).                            |
| `TOP_K`                |                              `5`                              | The maximum number of chunks retrieved per query. It can be overridden per request through the `top_k` field of `/ask`.                                                          |

### Generation parameters

Two generation parameters are defined as constants in [`app/composition.py`](app/composition.py) rather than as environment variables, because they express a design decision about how the system should behave rather than a deployment choice.

| Constant                 | Value  | Rationale                                                                                                                             |
| :----------------------- | :----: | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `GENERATION_TEMPERATURE` |  `0.1` | Produces near-deterministic output, which suits an extraction and synthesis task constrained by sources.                               |
| `MAX_OUTPUT_TOKENS`      |  `800` | Caps the maximum answer length and, as a direct consequence, the potential cost of every request.                                      |
