# ContractIQ

**AI-powered contract analysis platform — extract obligations, flag risks, and summarize legal documents in seconds.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat)](https://langchain.com)
[![AWS ECS](https://img.shields.io/badge/AWS-ECS%20Fargate-FF9900?style=flat&logo=amazonaws&logoColor=white)](https://aws.amazon.com/ecs)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is ContractIQ?

Freelancers, founders, and small businesses sign contracts without a lawyer. ContractIQ solves this — upload any legal document (NDA, employment agreement, vendor contract) and get:

- **Obligation extraction** — every commitment, deadline, and party responsibility pulled out and structured
- **Risk flagging** — unusual or one-sided clauses identified with plain-English explanations
- **Document comparison** — side-by-side diff between two contract versions, page by page
- **Conversational QnA** — ask questions about your contract in plain English

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client (HTML/JS)                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│              FastAPI Backend (api/main.py)               │
│   /analyze  │  /compare  │  /chat/index  │  /chat/query │
└──────┬───────────────────────────────┬───────────────────┘
       │                               │
┌──────▼──────┐                ┌───────▼──────────┐
│  LLM Chain  │                │   FAISS Vector   │
│  (Groq /    │                │   Store          │
│   Gemini)   │                │  (Session-based) │
└──────┬──────┘                └───────┬──────────┘
       │                               │
┌──────▼───────────────────────────────▼──────────┐
│              LangChain LCEL Pipeline             │
│  Prompt → LLM → OutputFixingParser → Response   │
└──────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│           AWS ECS Fargate (Production)           │
│  ECR Image │ Secrets Manager │ CloudWatch Logs  │
└─────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| Document Analysis | Extract metadata, summary, sentiment, and structured obligations from any PDF |
| Document Comparison | Page-by-page diff between two contract versions using LLM |
| Multi-doc Chat | Upload multiple documents, ask questions across all of them |
| Session Management | Each user session gets isolated FAISS index and file storage |
| Structured Logging | JSON structured logs via structlog — every request traced |
| Custom Exception Handling | Full traceback with file, line, and context captured |
| Production Deployment | AWS ECS Fargate + ECR + CloudFormation IaC + Secrets Manager |
| CI/CD Pipeline | GitHub Actions — build, test, push to ECR, deploy to ECS |

---

## Tech Stack

**AI & LLM**
- LangChain 0.3 — LCEL pipeline, output parsers, chat history
- LangChain-Groq — DeepSeek R1 Distill LLaMA 70B (primary LLM)
- LangChain-Google-GenAI — Gemini 2.0 Flash + text-embedding-004
- FAISS — vector store with idempotent document ingestion and fingerprinting

**Backend**
- FastAPI 0.116 — async REST API with file upload support
- Pydantic — structured output validation (Metadata, SummaryResponse models)
- structlog — JSON structured logging with ISO timestamps
- PyMuPDF — PDF text extraction page by page

**DevOps & Cloud**
- Docker + .dockerignore — production-optimised container
- AWS ECS Fargate — serverless container deployment
- AWS ECR — private container registry
- AWS Secrets Manager — secure API key management
- AWS CloudFormation — full infrastructure as code (VPC, subnets, ECS cluster)
- GitHub Actions — CI/CD pipeline

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional for local)
- API keys: Groq, Google Gemini, LangChain (optional)

### 1. Clone and install

```bash
git clone https://github.com/bhavjeetsingh/CONTRACTIQ.git
cd CONTRACTIQ
conda create -p env python=3.10 -y
conda activate ./env
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Run with Docker

```bash
docker build -t contractiq .
docker run -p 8080:8080 --env-file .env contractiq
```

Open `http://localhost:8080` in your browser.

---

## API Reference

### `POST /analyze`
Upload a PDF and extract structured metadata, obligations, and summary.

```bash
curl -X POST "http://localhost:8080/analyze" \
  -F "file=@contract.pdf"
```

**Response:**
```json
{
  "Title": "Non-Disclosure Agreement",
  "Author": ["Acme Corp"],
  "Summary": ["Mutual NDA between two parties..."],
  "SentimentTone": "Formal",
  "PageCount": 4
}
```

### `POST /compare`
Compare two PDF versions side by side.

```bash
curl -X POST "http://localhost:8080/compare" \
  -F "reference=@contract_v1.pdf" \
  -F "actual=@contract_v2.pdf"
```

### `POST /chat/index`
Index one or more documents for conversational QnA.

```bash
curl -X POST "http://localhost:8080/chat/index" \
  -F "files=@contract.pdf" \
  -F "session_id=my-session"
```

### `POST /chat/query`
Ask a question about your indexed documents.

```bash
curl -X POST "http://localhost:8080/chat/query" \
  -F "question=What are my obligations under this NDA?" \
  -F "session_id=my-session"
```

### `GET /health`
```json
{ "status": "ok", "service": "document-portal" }
```

---

## Project Structure

```
CONTRACTIQ/
├── api/                    # FastAPI app and routes
│   └── main.py
├── src/
│   ├── document_ingestion/ # PDF save, read, FAISS indexing
│   ├── document_analyzer/  # LLM metadata extraction
│   ├── document_compare/   # LLM document comparison
│   └── document_chat/      # Conversational RAG pipeline
├── model/
│   └── models.py           # Pydantic output schemas
├── prompt/
│   └── prompt_library.py   # All LangChain prompt templates
├── config/
│   └── config.yaml         # LLM and embedding config
├── logger/
│   └── custom_logger.py    # structlog JSON logger
├── exception/
│   └── custom_exception.py # Full traceback exception handler
├── infrastructure/
│   └── document-portal-cf.yaml  # AWS CloudFormation IaC
├── utils/                  # Model loader, file I/O helpers
├── tests/                  # pytest test suite
├── notebook/               # Experiments and prototypes
├── .github/workflows/      # GitHub Actions CI/CD
├── Dockerfile
└── requirements.txt
```

---

## Deployment

Full infrastructure defined in `infrastructure/document-portal-cf.yaml`:
- VPC with 2 public subnets across availability zones
- ECS Cluster + Fargate task (1 vCPU, 8GB RAM)
- ECR repository with image scanning
- Secrets Manager integration for API keys
- CloudWatch logging

```bash
aws cloudformation deploy \
  --template-file infrastructure/document-portal-cf.yaml \
  --stack-name contractiq-stack \
  --parameter-overrides ImageUrl=<your-ecr-image-uri> \
  --capabilities CAPABILITY_NAMED_IAM
```

---

## Roadmap

- [ ] JWT authentication — user accounts and protected routes
- [ ] Redis caching — cache LLM responses to reduce latency and cost
- [ ] Celery workers — background document processing for large files
- [ ] SSE streaming — real-time token-by-token response streaming
- [ ] RAGAS evaluation — automated RAG quality scoring
- [ ] LangSmith tracing — full LLM call observability
- [ ] Rate limiting — per-user request throttling
- [ ] Contract obligation export as PDF/CSV report

---

## Author

**Bhavjeet Singh**
IIT Madras — BS Data Science and Programming
[LinkedIn](https://linkedin.com/in/bhavjeetsingh) · [GitHub](https://github.com/bhavjeetsingh) · bhavjeetsingh784@gmail.com
