# 🏥 Multiagent Radiology Report Generation

> Production-grade agentic AI system for automated radiology report generation with mandatory human-in-the-loop, explainable AI, and full GDPR/HIPAA compliance.

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-orange)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green)](https://fastapi.tiangolo.com)
[![GDPR](https://img.shields.io/badge/GDPR-compliant-brightgreen)](https://gdpr.eu)
[![HIPAA](https://img.shields.io/badge/HIPAA-compliant-brightgreen)](https://hhs.gov/hipaa)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Art.%2014%20compliant-blue)](https://artificialintelligenceact.eu)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-yellow)](https://wandb.ai)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Compliance](#compliance)
- [MLOps](#mlops)
- [Roadmap](#roadmap)

---

## Overview

This system takes a DICOM medical scan as input and produces a structured, validated radiology report — with a radiologist reviewing and approving every single report before it is finalized. No report leaves the system without human sign-off.

The pipeline combines specialized medical vision models, retrieval-augmented generation over medical literature, LangGraph orchestration with automatic retry logic, and a mandatory human-in-the-loop checkpoint that satisfies EU AI Act Article 14 requirements for high-risk AI systems.

### What makes this production-grade

- **Agentic multi-agent system** — 4 autonomous agents with tool use, reasoning loops, and state management via LangGraph
- **Mandatory human oversight** — every report pauses for radiologist review before finalization (EU AI Act Art. 14)
- **Explainable AI** — Grad-CAM heatmaps show exactly which image regions drove the model's findings
- **GDPR/HIPAA compliant** — PII stripped on ingest, anonymized IDs throughout, 90-day retention, right to erasure, full audit trail
- **RAG-grounded reports** — clinical context retrieved from medical literature via Qdrant vector search
- **Prior patient history** — MCP server exposes PostgreSQL reports to any AI client including Claude Desktop
- **Full MLOps** — every pipeline run tracked in Weights & Biases with QA scores, latency, model versions
- **Production infrastructure** — FastAPI + PostgreSQL + Docker + GitHub Actions CI/CD + Terraform IaC for AWS

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Gradio UI (port 7860)                       │
│   DICOM / PNG upload · Clinical note · Scan viewer · HIL panel  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend (port 8000)                     │
│   Reports CRUD · Pipeline trigger · GDPR endpoints · /metrics   │
│   medical-ai-middleware: rate limiting · consent · sec headers   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    DICOM Pipeline                               │
│   pydicom load → strip PII → sha256 anonymized_id → 512x512 PNG │
│   TorchXRayVision (background thread) → Grad-CAM heatmap        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              LangGraph Orchestrator (StateGraph)                 │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   Agent 1    │──▶│   Agent 2    │──▶│   Agent 3    │        │
│  │Image Analysis│   │  Clinical    │   │   Report     │        │
│  │              │   │  Context     │   │  Drafting    │        │
│  │ Vision LLM   │   │ Qdrant RAG   │   │    LLM       │        │
│  │ + findings   │   │ + MCP prior  │   │  + context   │        │
│  └──────────────┘   │   reports    │   └──────┬───────┘        │
│                     └──────────────┘          │                 │
│                                        ┌──────▼───────┐        │
│                                        │   Agent 4    │        │
│                                        │ QA Validation│        │
│                                        │ completeness │        │
│                                        │ consistency  │        │
│                                        │ hallucination│        │
│                                        └──────┬───────┘        │
│                                               │                 │
│                                    ┌──────────▼──────────┐     │
│                                    │  Human Review (HIL) │     │
│                                    │  graph.interrupt()  │     │
│                                    │  MANDATORY — always │     │
│                                    └──────────┬──────────┘     │
└───────────────────────────────────────────────┼────────────────┘
                                                │
┌───────────────────────────────────────────────▼────────────────┐
│                     Data Layer                                  │
│   PostgreSQL (reports + audit log) · Qdrant (embeddings)        │
│   S3 (DICOM/PNG storage) · SQLite (LangGraph checkpoints)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

### Agent 1 — Image Analysis

Receives the anonymized PNG scan and sends it to a vision-language model for structured finding extraction.

**Current:** Groq Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`) — free tier, 300+ tokens/sec  
**Production:** Google MedGemma 4b via Vertex AI — trained specifically on medical imaging including radiology, pathology, dermatology, and ophthalmology

**Output:** Structured `ImageFindings` with findings list, impression, confidence score, and urgency flag.

**XAI:** TorchXRayVision DenseNet (densenet121-res224-all, trained on CheXpert + NIH + MIMIC + PadChest) runs in parallel, scoring 18 chest pathologies and generating Grad-CAM heatmaps via `medical-ai-middleware`. When MedGemma is available, attention maps replace Grad-CAM using `AttentionMap(model, model_type="medgemma")` from the same middleware.

### Agent 2 — Clinical Context

Retrieves relevant medical knowledge from two sources simultaneously:

1. **Qdrant vector search** — semantic search over curated medical literature (radiology guidelines, differential diagnoses, follow-up recommendations)
2. **MCP server** — queries PostgreSQL for prior approved reports for the same patient, enabling longitudinal comparison

The radiologist's clinical note (e.g. "58yo male, smoker, chest pain 3 days, rule out PE") is prepended to the search query, significantly improving retrieval relevance.

**Output:** `ClinicalContext` with conditions, differential diagnosis, follow-up recommendations, prior reports summary, and urgency level.

### Agent 3 — Report Drafting

Takes image findings + clinical context + clinical note and generates a structured radiology report following standard format:

```
CLINICAL INDICATION
TECHNIQUE
FINDINGS
IMPRESSION
RECOMMENDATIONS
```

**Current:** Groq Llama 4 Scout  
**Production:** Anthropic Claude Sonnet — lowest hallucination rate, best structured medical writing, HIPAA BAA available

### Agent 4 — QA Validation

Reviews the drafted report against the original image findings using both rule-based checks and LLM semantic validation:

- **Completeness check** — all 5 required sections present and non-empty
- **Urgency check** — critical keywords detected with negation awareness (handles "no pneumothorax" correctly)
- **Consistency check** — report findings match image analysis impression
- **Hallucination check** — LLM verifies report claims are supported by image evidence

If QA fails, LangGraph automatically routes back to Agent 3 for re-drafting (max 3 retries). If max retries hit, sends to human review with QA issues noted.

**Current QA score: 0.9**

### Human Review (Mandatory HIL)

After QA passes, `graph.interrupt()` pauses the pipeline and saves full state to SQLite. The radiologist sees:

- Original scan image
- Grad-CAM heatmap overlay
- AI-generated report (fully editable)
- QA score and any issues
- Whether prior reports were found
- Urgency level

The radiologist can edit the report directly before approving. No report is finalized without explicit approval. This satisfies:

- EU AI Act Article 14 — human oversight for high-risk AI
- Clinical governance — accountability for medical decisions
- HIPAA — human accountability for medical record creation

---

## Tech Stack

### AI / Agents
| Component | Current | Production |
|-----------|---------|------------|
| Vision model | Groq Llama 4 Scout | Google MedGemma 4b (Vertex AI) |
| Report generation | Groq Llama 4 Scout | Anthropic Claude Sonnet |
| QA validation | Groq Llama 4 Scout | Anthropic Claude Sonnet |
| XAI (chest) | TorchXRayVision + Grad-CAM | TorchXRayVision + Grad-CAM |
| XAI (other modalities) | N/A | MedGemma attention maps |
| Agent orchestration | LangGraph 1.1 | LangGraph 1.1 |
| Vector search | Qdrant (local) | Qdrant Cloud |

### Backend
| Component | Technology |
|-----------|------------|
| API framework | FastAPI 0.135 |
| Database | PostgreSQL 16 (RDS in prod) |
| Task queue | Celery + Redis |
| DICOM processing | pydicom + Pillow |
| Medical imaging | TorchXRayVision, MONAI |
| XAI middleware | medical-ai-middleware (custom) |
| MCP server | Python MCP SDK |

### Infrastructure
| Component | Technology |
|-----------|------------|
| Containerization | Docker + Docker Compose |
| Cloud (IaC) | Terraform — ECS + RDS + S3 + ECR |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Experiment tracking | Weights & Biases |
| Object storage | AWS S3 (EU region — GDPR) |

### Compliance
| Requirement | Implementation |
|-------------|----------------|
| GDPR Art. 5 | 90-day data retention, auto-delete |
| GDPR Art. 17 | Right to erasure endpoint |
| GDPR Art. 25 | PII stripped on DICOM ingest |
| HIPAA | 6-year audit log retention |
| EU AI Act Art. 14 | Mandatory human review node |
| Security | Rate limiting, consent enforcement, security headers via medical-ai-middleware |

---

## Features

- **Multi-modal input** — DICOM files, PNG/JPG images, clinical notes
- **DICOM anonymization** — PatientName, PatientID, DOB, and 10+ PII fields stripped automatically
- **Grad-CAM heatmaps** — visual explanation of model attention on chest scans
- **RAG clinical context** — Qdrant semantic search over medical literature
- **Prior report retrieval** — MCP server exposes PostgreSQL to AI clients including Claude Desktop
- **Mandatory HIL** — every report paused for radiologist review, full edit capability
- **Retry logic** — automatic re-drafting on QA failure (max 3 attempts)
- **Audit trail** — every action logged to PostgreSQL (HIPAA compliant)
- **Right to erasure** — GDPR Article 17 endpoint deletes all data for a patient
- **W&B tracking** — QA scores, latency, model versions, urgency distribution per run
- **Prometheus metrics** — request count, inference duration, error rates
- **Grafana dashboard** — real-time pipeline monitoring

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Groq API key (free at console.groq.com)
- Weights & Biases account (free at wandb.ai)

### 1. Clone and install

```bash
git clone https://github.com/moebouassida/multiagent-radiology-report.git
cd multiagent-radiology-report

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install "medical-ai-middleware[all] @ git+https://github.com/moebouassida/medical-ai-middleware.git"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:
```bash
GROQ_API_KEY=your-groq-key
WANDB_API_KEY=your-wandb-key
DATABASE_URL=postgresql://radiology:password@localhost:5432/radiology_db
```

### 3. Start services

```bash
# Start PostgreSQL + Qdrant
docker compose up postgres qdrant -d

# Ingest medical knowledge into Qdrant
python mlops/ingest_medical_knowledge.py
```

### 4. Run the system

```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Gradio UI
python ui/app.py
```

Open **http://localhost:7860** in your browser.

### 5. Run a scan

1. Upload a DICOM file (`.dcm`)
2. Select modality (CR, MR, CT, DX)
3. Optionally add a clinical note — e.g. `58yo male, chest pain, rule out PE`
4. Click **Analyze Scan**
5. Review the AI-generated report
6. Edit if needed, then click **Approve & Finalize**

The approved report is saved to PostgreSQL with a full audit trail.

---

## Configuration

All configuration via environment variables. See `.env.example` for full list.

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for vision + report generation | required |
| `GROQ_MODEL` | Groq model name | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `DATABASE_URL` | PostgreSQL connection string | required |
| `QDRANT_URL` | Qdrant vector DB URL | `http://localhost:6333` |
| `WANDB_API_KEY` | Weights & Biases API key | optional |
| `WANDB_PROJECT` | W&B project name | `radiology-ai` |
| `DATA_RETENTION_DAYS` | GDPR retention period | `90` |
| `OLLAMA_BASE_URL` | Local Ollama URL (dev fallback) | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Local model name | `qwen3.5:4b-q4_K_M` |

### Model configuration

The system selects the inference backend automatically based on available API keys:

```
Priority: GROQ_API_KEY → OLLAMA_MODEL → OPENROUTER_API_KEY → mock mode
```

---

## API Reference

The FastAPI backend exposes a full REST API at `http://localhost:8000/docs`.

### Core endpoints

```
POST /pipeline/analyze          Upload DICOM + run full pipeline
GET  /reports/{report_id}       Get a specific report
GET  /reports/scan/{anon_id}    Get all reports for a patient
POST /reports/{id}/approve      Approve a report (radiologist)
POST /reports/{id}/reject       Reject a report
GET  /reports/                  List reports with filters
```

### Compliance endpoints

```
GET  /compliance/report         GDPR/HIPAA compliance summary
POST /compliance/retention/cleanup   Run data retention cleanup
DELETE /compliance/erase/{anon_id}   GDPR right to erasure
```

### Health endpoints

```
GET /health                     API health check
GET /health/db                  Database connectivity check
GET /metrics                    Prometheus metrics
```

### MCP server (Claude Desktop integration)

The MCP server allows any MCP-compatible client (Claude Desktop, custom agents) to query the radiology database using natural language:

```bash
python mcp_server/radiology_mcp.py
```

Available tools:
- `get_prior_reports(anonymized_id)` — fetch patient's report history
- `get_report_by_id(report_id)` — fetch specific report
- `search_reports(query, modality)` — keyword search across reports
- `get_patient_summary(anonymized_id)` — full patient radiological history

Configure Claude Desktop by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "radiology-reports": {
      "command": "python",
      "args": ["/path/to/mcp_server/radiology_mcp.py"]
    }
  }
}
```

---

## Compliance

### GDPR

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 4(1) | Anonymization | PatientName, PatientID, DOB, address, phone stripped on DICOM ingest. SHA-256 hashed anonymous ID used throughout. |
| Art. 5(1)(e) | Data minimization | Only clinical metadata stored. Raw DICOM deleted after processing. |
| Art. 5(1)(e) | Storage limitation | Reports auto-deleted after 90 days via `POST /compliance/retention/cleanup` |
| Art. 17 | Right to erasure | `DELETE /compliance/erase/{anonymized_id}` removes all reports for a patient |
| Art. 25 | Privacy by design | PII never reaches LLM. Only pixel data + safe metadata sent for inference. |
| Art. 32 | Security | HTTPS, security headers, rate limiting, IP anonymization via medical-ai-middleware |

### HIPAA

| Requirement | Implementation |
|-------------|----------------|
| Audit controls | Every action logged to `audit_log` table with timestamp, user, IP, action type |
| Audit retention | Audit logs retained for 6 years (2190 days) |
| Access controls | JWT authentication on all API endpoints |
| Integrity | Human approval required before report finalization |
| Transmission security | TLS enforced, security headers via middleware |

### EU AI Act

Radiology AI is explicitly classified as **high-risk** under EU AI Act Annex III. This system implements:

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 14 | Human oversight | `graph.interrupt()` — every report pauses for mandatory radiologist review |
| Art. 13 | Transparency | XAI heatmaps show model attention. QA scores displayed to radiologist. |
| Art. 9 | Risk management | QA validation agent checks completeness, consistency, urgency flags |
| Art. 12 | Record keeping | Full audit trail in PostgreSQL. W&B experiment tracking. |

---

## MLOps

Every pipeline run is automatically tracked in Weights & Biases:

**Metrics tracked per run:**
- QA score (0.0 — 1.0)
- Pipeline latency (seconds)
- Retry count
- Findings count
- Urgency level
- Human approved (boolean)
- Model name and version

**View your runs:** https://wandb.ai/moebouassida-soci-t-g-n-rale/radiology-ai

### Prometheus metrics

Available at `GET /metrics`:

- `http_requests_total` — request count by endpoint
- `http_request_duration_seconds` — latency histogram
- `inference_duration_seconds` — model inference time
- `inference_requests_total` — inference count by model

### Running Grafana dashboard

```bash
docker compose up prometheus grafana -d
```

Open **http://localhost:3000** (admin/admin), import the dashboard from `infra/grafana_dashboard.json`.

---

## Roadmap

### In progress
- [ ] MedGemma 4b integration (RTX 2060 / Google Vertex AI)
- [ ] Claude Sonnet for report generation and QA
- [ ] AWS deployment (ECS + RDS + S3 + ECR via Terraform)
- [ ] MIMIC-CXR dataset integration for proper chest X-ray testing

### Planned
- [ ] Multi-frame DICOM support (CT/MRI series)
- [ ] DICOM metadata extraction (slice thickness, TR/TE, KVP) as model context
- [ ] Structured report export (HL7 FHIR R4)
- [ ] Fine-tuning pipeline on MIMIC-CXR reports
- [ ] Multi-radiologist consensus mode
- [ ] MedGemma attention maps for non-chest modalities

### Model upgrade path

```
Current (development)          Production
──────────────────────         ──────────
Groq Llama 4 Scout       →     MedGemma 4b (Vertex AI, HIPAA BAA)
  vision + findings              medical specialist vision model
  
Groq Llama 4 Scout       →     Claude Sonnet (Anthropic API, HIPAA BAA)
  report + QA                    best structured medical writing
  
TorchXRayVision          →     TorchXRayVision (keep)
  chest detection + CAM          + MedGemma attention maps for other modalities
```

The architecture is model-agnostic — swapping models requires changing 2-3 environment variables, no code changes.

---

## Project Structure

```
multiagent-radiology-report/
├── agents/                    # LangGraph agents
│   ├── image_analysis.py      # Vision model → structured findings
│   ├── clinical_context.py    # Qdrant RAG + MCP prior reports
│   ├── report_drafting.py     # LLM report generation
│   ├── qa_validation.py       # Completeness + consistency checks
│   └── orchestrator.py        # LangGraph StateGraph + HIL
├── pipeline/                  # DICOM processing
│   ├── dicom_loader.py        # Load + anonymize DICOM
│   ├── preprocessor.py        # Normalize + convert to PNG
│   └── xai.py                 # Grad-CAM via medical-ai-middleware
├── api/                       # FastAPI backend
│   ├── main.py                # App entry point + middleware
│   ├── compliance.py          # GDPR/HIPAA logic
│   ├── models/                # SQLAlchemy models (Report, AuditLog)
│   └── routes/                # REST endpoints
├── mcp_server/                # MCP server for Claude Desktop
│   └── radiology_mcp.py       # Exposes PostgreSQL as MCP tools
├── mlops/                     # MLOps
│   ├── tracking.py            # W&B experiment tracking
│   └── ingest_medical_knowledge.py  # Populate Qdrant
├── ui/                        # Gradio frontend
│   └── app.py                 # Radiologist dashboard
├── infra/                     # Infrastructure as code
│   ├── main.tf                # AWS Terraform (ECS + RDS + S3 + ECR)
│   └── prometheus.yml         # Prometheus scrape config
├── tests/                     # Test suite
├── docker-compose.yml         # Local development stack
├── Dockerfile                 # Production container
└── pyproject.toml             # Dependencies
```

---

## Author

**Moez Bouassida** — AI/ML Engineer · Medical Imaging  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/moezbouassida/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/moebouassida)
[![Medium](https://img.shields.io/badge/Medium-Read-green)](https://medium.com/@moezbouassida)

---

## Related Projects

- [medical-ai-middleware](https://github.com/moebouassida/medical-ai-middleware) — GDPR compliance, Prometheus monitoring, XAI (Grad-CAM + attention maps) for medical AI APIs
- [SwinUNETR-3D-Brain-Segmentation](https://github.com/moebouassida/SwinUNETR-3D-Brain-Segmentation) — 3D brain tumor segmentation
- [Path-VQA-Med-GaMMa-Fine-Tuning](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning) — MedGemma fine-tuning on pathology VQA
- [Breast-Cancer-Segmentation](https://github.com/moebouassida/Breast-Cancer-Segmentation) — U-Net breast cancer segmentation

---

*This system is an AI assistant for qualified radiologists. All reports must be reviewed and approved by a licensed radiologist before clinical use. Not intended for direct clinical decision-making without human oversight.*