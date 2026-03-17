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

## Demo

<p align="center">
  <img src="docs/demo.gif" alt="Radiology AI Demo" width="800"/>
</p>

### Grad-CAM XAI Heatmaps — Real Chest X-Rays

The system highlights exactly which regions drove the model's findings. Red/yellow = high attention, original image shows through elsewhere.

| Chest X-Ray + Grad-CAM | Clinical finding |
|---|---|
| ![Heatmap 1](docs/heatmap1.png) | Left upper lobe opacity — model correctly identifies abnormal region |
| ![Heatmap 2](docs/heatmap2.png) | Left mid-zone consolidation — model focuses on lung parenchyma, ignores hardware |

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
- **Explainable AI** — Grad-CAM heatmaps via [medical-ai-middleware](https://github.com/moebouassida/medical-ai-middleware) show exactly which image regions drove the model's findings
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
│  DICOM/PNG upload · Clinical note · Scan viewer · HIL panel     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend (port 8000)                     │
│   Reports CRUD · Pipeline trigger · GDPR endpoints · /metrics   │
│   medical-ai-middleware: rate limiting · consent · sec headers   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      DICOM Pipeline                             │
│  pydicom load → strip PII → sha256 anon_id → 512x512 PNG        │
│  TorchXRayVision (background thread) → Grad-CAM heatmap         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              LangGraph Orchestrator (StateGraph)                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Agent 1    │─▶│   Agent 2    │─▶│   Agent 3    │          │
│  │Image Analysis│  │  Clinical    │  │   Report     │          │
│  │ Groq vision  │  │  Context     │  │  Drafting    │          │
│  │ + findings   │  │ Qdrant RAG   │  │  Groq LLM    │          │
│  └──────────────┘  │ + MCP prior  │  └──────┬───────┘          │
│                    │   reports    │         │                   │
│                    └──────────────┘  ┌──────▼───────┐          │
│                                      │   Agent 4    │          │
│                                      │ QA Validation│          │
│                                      │ score: 0.9   │          │
│                                      └──────┬───────┘          │
│                                             │                   │
│                                  ┌──────────▼──────────┐       │
│                                  │  Human Review (HIL) │       │
│                                  │  graph.interrupt()  │       │
│                                  │  MANDATORY — always │       │
│                                  └──────────┬──────────┘       │
└─────────────────────────────────────────────┼──────────────────┘
                                              │
┌─────────────────────────────────────────────▼──────────────────┐
│                        Data Layer                               │
│  PostgreSQL (reports + audit log) · Qdrant (embeddings)         │
│  S3 (DICOM/PNG storage) · SQLite (LangGraph checkpoints)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

### Agent 1 — Image Analysis

Receives the anonymized PNG scan and sends it to a vision-language model for structured finding extraction.

**Current:** Groq Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`) — free tier, 300+ tokens/sec, vision capable  
**Production:** Google MedGemma 4b via Vertex AI — trained specifically on medical imaging including radiology, pathology, dermatology, and ophthalmology

**Output:** Structured `ImageFindings` dataclass with findings list, impression, confidence score, and urgency flag.

**XAI:** TorchXRayVision DenseNet (densenet121-res224-all, trained on CheXpert + NIH + MIMIC + PadChest) runs in a background thread simultaneously, scoring 18 chest pathologies and generating Grad-CAM heatmaps via [medical-ai-middleware](https://github.com/moebouassida/medical-ai-middleware). XAI only activates for chest modalities (CR, DX) — skipped for MRI/CT where TorchXRayVision is not applicable. When MedGemma is available, attention maps replace Grad-CAM for all modalities using `AttentionMap(model, model_type="medgemma")`.

### Agent 2 — Clinical Context

Retrieves relevant medical knowledge from two sources simultaneously:

1. **Qdrant vector search** — semantic search over curated medical literature (radiology guidelines, differential diagnoses, follow-up recommendations)
2. **MCP server** — queries PostgreSQL for prior approved reports for the same patient, enabling longitudinal comparison

The radiologist's clinical note (e.g. `58yo male, smoker, chest pain 3 days, rule out PE`) is prepended to the Qdrant search query, significantly improving retrieval relevance.

**Output:** `ClinicalContext` dataclass with conditions, differential diagnosis, follow-up recommendations, prior reports summary, and urgency level.

### Agent 3 — Report Drafting

Takes image findings + clinical context + clinical note and generates a structured radiology report:

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

Reviews the drafted report using both rule-based checks and LLM semantic validation:

- **Completeness** — all 5 required sections present and non-empty
- **Urgency** — critical keywords detected with negation awareness (`no pneumothorax` correctly handled)
- **Consistency** — report findings match image analysis impression
- **Hallucination** — LLM verifies report claims are supported by image evidence

If QA fails, LangGraph automatically routes back to Agent 3 for re-drafting (max 3 retries). After max retries, sends to human review with QA issues noted. **Current QA score: 0.9**

### Human Review — Mandatory HIL

After QA passes, `graph.interrupt()` pauses the pipeline and saves full state to SQLite. The radiologist sees the scan, heatmap, AI report (fully editable), QA score, and urgency level. No report is finalized without explicit approval.

This satisfies EU AI Act Article 14, clinical governance, and HIPAA human accountability requirements.

---

## Tech Stack

### AI / Agents

| Component | Current | Production |
|-----------|---------|------------|
| Vision model | Groq Llama 4 Scout | Google MedGemma 4b (Vertex AI) |
| Report generation | Groq Llama 4 Scout | Anthropic Claude Sonnet |
| QA validation | Groq Llama 4 Scout | Anthropic Claude Sonnet |
| XAI — chest | TorchXRayVision + Grad-CAM | TorchXRayVision + Grad-CAM |
| XAI — other modalities | N/A | MedGemma attention maps |
| Agent orchestration | LangGraph 1.1 | LangGraph 1.1 |
| Vector search | Qdrant local | Qdrant Cloud |

### Backend

| Component | Technology |
|-----------|------------|
| API framework | FastAPI 0.135 |
| Database | PostgreSQL 16 (RDS in prod) |
| DICOM processing | pydicom + Pillow |
| Medical imaging | TorchXRayVision |
| XAI middleware | [medical-ai-middleware](https://github.com/moebouassida/medical-ai-middleware) |
| MCP server | Python MCP SDK |
| Compliance | medical-ai-middleware (GDPR + rate limiting + security headers) |

### Infrastructure

| Component | Technology |
|-----------|------------|
| Containerization | Docker + Docker Compose |
| Cloud IaC | Terraform — ECS + RDS + S3 + ECR |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Experiment tracking | Weights & Biases |
| Object storage | AWS S3 (eu-west-1 — GDPR data residency) |

---

## Features

- **Multi-modal input** — DICOM files, PNG/JPG images, free-text clinical notes
- **DICOM anonymization** — PatientName, PatientID, DOB, and 10+ PII fields stripped on ingest
- **Grad-CAM heatmaps** — clean overlay showing model attention on chest scans (see demo above)
- **RAG clinical context** — Qdrant semantic search over 20 curated medical knowledge entries
- **Prior report retrieval** — MCP server exposes PostgreSQL to Claude Desktop and other AI clients
- **Mandatory HIL** — every report paused for radiologist review regardless of QA score
- **Retry logic** — automatic re-drafting on QA failure (max 3 attempts)
- **Audit trail** — every action logged to PostgreSQL with timestamp, IP, user (HIPAA)
- **Right to erasure** — GDPR Article 17 endpoint
- **W&B tracking** — QA scores, latency, model versions per run
- **Prometheus metrics** — request count, inference time, error rates
- **Terraform IaC** — full AWS infrastructure as code, deploy-ready

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (for PostgreSQL + Qdrant)
- [Groq API key](https://console.groq.com) — free tier, no credit card
- [Weights & Biases account](https://wandb.ai) — free tier

### 1. Clone and install

```bash
git clone https://github.com/moebouassida/multiagent-radiology-report.git
cd multiagent-radiology-report

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pip install "medical-ai-middleware[all] @ git+https://github.com/moebouassida/medical-ai-middleware.git"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Minimum required variables:

```bash
GROQ_API_KEY=your-groq-key
WANDB_API_KEY=your-wandb-key
DATABASE_URL=postgresql://radiology:password@localhost:5432/radiology_db
QDRANT_URL=http://localhost:6333
```

### 3. Start infrastructure

```bash
# PostgreSQL + Qdrant
docker compose up postgres qdrant -d

# Populate Qdrant with medical knowledge
python mlops/ingest_medical_knowledge.py
```

### 4. Run

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — UI
python ui/app.py
```

Open **http://localhost:7860**

### 5. Test the pipeline

1. Upload a chest X-ray DICOM (`.dcm`) — [get sample data from Kaggle](https://www.kaggle.com/datasets/falahgatea/chest-x-ray-dicom)
2. Select modality **CR**
3. Add a clinical note: `65yo male smoker, productive cough 2 weeks, rule out pneumonia`
4. Click **Analyze Scan**
5. Review the report + Grad-CAM heatmap
6. Edit if needed → **Approve & Finalize**

---

## Configuration

All settings via environment variables. Full reference in `.env.example`.

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Groq model | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `DATABASE_URL` | PostgreSQL URL | required |
| `QDRANT_URL` | Qdrant URL | `http://localhost:6333` |
| `WANDB_API_KEY` | W&B API key | optional |
| `DATA_RETENTION_DAYS` | GDPR retention | `90` |
| `OLLAMA_BASE_URL` | Local Ollama (fallback) | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Local model (fallback) | `qwen3.5:4b-q4_K_M` |

**Inference backend priority:**
```
GROQ_API_KEY set  →  Groq cloud (recommended)
OLLAMA_MODEL set  →  local Ollama
OPENROUTER_API_KEY set  →  OpenRouter
none  →  mock mode (for testing)
```

---

## API Reference

Full interactive docs at **http://localhost:8000/docs**

### Pipeline

```
POST /pipeline/analyze     Upload DICOM + run full 4-agent pipeline
```

### Reports

```
GET    /reports/                    List reports (filterable by urgency, approved)
GET    /reports/{id}                Get specific report
GET    /reports/scan/{anon_id}      All reports for a patient
POST   /reports/{id}/approve        Radiologist approves report
POST   /reports/{id}/reject         Radiologist rejects report
```

### GDPR / Compliance

```
GET    /compliance/report                   Compliance summary for DPO
POST   /compliance/retention/cleanup        Run 90-day retention cleanup
DELETE /compliance/erase/{anonymized_id}    GDPR Art. 17 right to erasure
```

### Health

```
GET /health      API status
GET /health/db   Database connectivity
GET /metrics     Prometheus metrics
```

### MCP Server — Claude Desktop Integration

The MCP server lets any MCP-compatible client (Claude Desktop, custom agents) query your radiology database conversationally.

```bash
# start MCP server
python mcp_server/radiology_mcp.py
```

Available tools: `get_prior_reports`, `get_report_by_id`, `search_reports`, `get_patient_summary`

Add to Claude Desktop config:

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

You can then ask Claude: *"What were the findings for the last chest X-ray? Were there any urgent cases this week?"*

---

## Compliance

### GDPR

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 4(1) | Anonymization | 12 PII DICOM tags stripped on ingest. SHA-256 hashed anonymous ID throughout. PII never reaches LLM. |
| Art. 5(1)(e) | Storage limitation | Reports auto-deleted after 90 days |
| Art. 17 | Right to erasure | `DELETE /compliance/erase/{anonymized_id}` |
| Art. 25 | Privacy by design | Only pixel data + safe metadata sent for inference |
| Art. 32 | Security | HTTPS, security headers, rate limiting, IP anonymization via medical-ai-middleware |

### HIPAA

| Requirement | Implementation |
|-------------|----------------|
| Audit controls | Every action logged to `audit_log` table (timestamp, user, IP, action) |
| Audit retention | Logs kept 6 years (2190 days) |
| Integrity | Human approval required — no report finalized without radiologist sign-off |
| Transmission security | TLS enforced, security headers |

### EU AI Act

Radiology AI is **high-risk** under EU AI Act Annex III. This system implements:

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 14 | Human oversight | `graph.interrupt()` — mandatory radiologist review on every report |
| Art. 13 | Transparency | Grad-CAM heatmaps + QA scores shown to radiologist |
| Art. 9 | Risk management | QA validation agent with completeness, consistency, urgency checks |
| Art. 12 | Record keeping | Full audit trail in PostgreSQL + W&B experiment tracking |

---

## MLOps

Every pipeline run is automatically tracked in Weights & Biases.

**Tracked per run:** QA score · latency · retry count · findings count · urgency level · human approved · model name

**View live runs:** https://wandb.ai/moebouassida-soci-t-g-n-rale/radiology-ai

### Prometheus

Available at `GET /metrics`:
- `http_requests_total` — by endpoint/method/status
- `http_request_duration_seconds` — latency histogram
- `inference_duration_seconds` — model inference time

### Grafana

```bash
docker compose up prometheus grafana -d
# open http://localhost:3000 (admin/admin)
```

---

## Roadmap

### In progress
- [ ] MedGemma 4b (RTX 2060 / Google Vertex AI) — replaces Groq for vision
- [ ] Claude Sonnet — replaces Groq for report generation and QA
- [ ] AWS deployment via Terraform (ECS + RDS + S3 + ECR)
- [ ] MIMIC-CXR dataset (227k chest X-rays) for proper benchmarking

### Planned
- [ ] Multi-frame DICOM support (CT/MRI series)
- [ ] DICOM acquisition metadata as model context (slice thickness, TR/TE, KVP)
- [ ] HL7 FHIR R4 structured report export
- [ ] Fine-tuning on MIMIC-CXR radiology reports
- [ ] Multi-radiologist consensus mode
- [ ] MedGemma attention maps for non-chest modalities (MRI, CT, pathology)

### Production model upgrade (zero code changes needed)

```
Current                        Production
───────                        ──────────
Groq Llama 4 Scout  →  MedGemma 4b (Vertex AI, HIPAA BAA)
  image analysis         medical specialist, all modalities

Groq Llama 4 Scout  →  Claude Sonnet (Anthropic API, HIPAA BAA)
  report + QA            lowest hallucination, best medical writing

TorchXRayVision     →  TorchXRayVision (keep for chest)
  + Grad-CAM             + MedGemma attention maps (all modalities)
```

Swapping models = changing 2 environment variables.

---

## Project Structure

```
multiagent-radiology-report/
├── agents/
│   ├── image_analysis.py      # Vision LLM → structured findings
│   ├── clinical_context.py    # Qdrant RAG + MCP prior reports
│   ├── report_drafting.py     # LLM report generation
│   ├── qa_validation.py       # Rule-based + LLM validation
│   └── orchestrator.py        # LangGraph StateGraph + HIL
├── pipeline/
│   ├── dicom_loader.py        # Load + strip PII
│   ├── preprocessor.py        # Normalize → 512x512 PNG
│   └── xai.py                 # Grad-CAM via medical-ai-middleware
├── api/
│   ├── main.py                # FastAPI + middleware setup
│   ├── compliance.py          # GDPR/HIPAA logic
│   ├── models/                # SQLAlchemy (Report, AuditLog)
│   └── routes/                # REST endpoints
├── mcp_server/
│   └── radiology_mcp.py       # MCP tools over PostgreSQL
├── mlops/
│   ├── tracking.py            # W&B tracking
│   └── ingest_medical_knowledge.py
├── ui/
│   └── app.py                 # Gradio radiologist dashboard
├── infra/
│   ├── main.tf                # AWS Terraform
│   ├── variables.tf
│   ├── outputs.tf
│   └── prometheus.yml
├── docs/
│   ├── demo.gif               # Full pipeline demo
│   ├── heatmap_demo1.png      # Grad-CAM chest X-ray 1
│   └── heatmap_demo2.png      # Grad-CAM chest X-ray 2
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Author

**Moez Bouassida** — AI/ML Engineer · Medical Imaging

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/moezbouassida/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/moebouassida)
[![Medium](https://img.shields.io/badge/Medium-Read-green)](https://medium.com/@moezbouassida)

---

## Related Projects

- [medical-ai-middleware](https://github.com/moebouassida/medical-ai-middleware) — GDPR, Prometheus monitoring, Grad-CAM + attention maps for medical AI APIs
- [SwinUNETR-3D-Brain-Segmentation](https://github.com/moebouassida/SwinUNETR-3D-Brain-Segmentation) — 3D brain tumor segmentation
- [Path-VQA-Med-GaMMa-Fine-Tuning](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning) — MedGemma fine-tuning on pathology VQA
- [Breast-Cancer-Segmentation](https://github.com/moebouassida/Breast-Cancer-Segmentation) — U-Net breast cancer segmentation

---

*AI assistant for qualified radiologists. All reports must be reviewed and approved by a licensed radiologist before clinical use.*