<div align="center">

<img src="./static/image/MiroFish_logo_compressed.jpeg" alt="MiroFish Logo" width="75%"/>

**A Swarm Intelligence Prediction Engine**

[![GitHub Stars](https://img.shields.io/github/stars/666ghj/MiroFish?style=flat-square&color=DAA520)](https://github.com/666ghj/MiroFish/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/666ghj/MiroFish?style=flat-square)](https://github.com/666ghj/MiroFish/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.com/channels/1469200078932545606/1469201282077163739)
[![X](https://img.shields.io/badge/X-Follow-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/mirofish_ai)

</div>

## Overview

MiroFish is a multi-agent simulation engine that predicts outcomes by constructing a parallel digital world from seed data. It ingests documents (news, reports, financial signals), builds a knowledge graph, spawns thousands of AI agents with distinct personas and long-term memory, runs dual-platform social media simulations, and generates structured prediction reports.

**Input**: Upload seed documents (PDF/MD/TXT) + describe your prediction scenario in natural language.

**Output**: A detailed prediction report + an interactive simulated world you can query.

## Architecture

```
MiroFish/
├── frontend/          Vue 3 + Vite + D3.js (port 3000)
├── backend/           Python 3.11 + Flask (port 5001)
│   ├── app/
│   │   ├── api/       REST endpoints (graph, simulation, report)
│   │   ├── services/  Core business logic
│   │   ├── models/    Data models (Project, Task, SimulationState)
│   │   ├── utils/     LLM client, file parser, logger, retry
│   │   └── config.py  Environment-driven configuration
│   └── uploads/       File-based persistence (projects, simulations, reports)
├── static/image/      Logo assets
├── .env.example       Configuration template
├── Dockerfile         Single-container build (Node + Python)
└── docker-compose.yml Production deployment
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Vue 3, Vue Router, Axios, D3.js | SPA with knowledge graph visualization |
| **Backend** | Flask 3.0, Flask-CORS | REST API server |
| **LLM** | OpenAI SDK (any compatible provider) | Ontology generation, persona creation, report writing |
| **Memory** | Zep Cloud (GraphRAG) | Knowledge graph storage, entity extraction, semantic search |
| **Simulation** | CAMEL-AI OASIS | Multi-agent social media simulation (Twitter + Reddit) |
| **File Processing** | PyMuPDF, charset-normalizer | PDF/MD/TXT extraction with encoding detection |
| **Package Management** | uv (Python), npm (Node) | Dependency management |

### Data Flow

```
Documents + Requirement
        │
        ▼
┌─ Step 1: Graph Building ─────────────────────────┐
│  FileParser → TextProcessor → OntologyGenerator   │
│  (LLM generates entity/edge types)                │
│  → GraphBuilderService → Zep Cloud (GraphRAG)     │
└───────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 2: Environment Setup ───────────────────────┐
│  ZepEntityReader → extracts entities from graph    │
│  OasisProfileGenerator → LLM creates agent personas│
│  SimulationConfigGenerator → LLM plans simulation  │
└───────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 3: Simulation ─────────────────────────────┐
│  SimulationRunner → spawns subprocess             │
│  OASIS Twitter + Reddit run in parallel           │
│  ZepGraphMemoryManager → updates graph per round  │
│  IPC protocol for real-time status & interviews   │
└───────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 4: Report Generation ──────────────────────┐
│  ReportAgent (ReACT pattern with tool use)        │
│  Tools: insight_forge, panorama_search,           │
│         quick_search, interview_agents            │
│  Generates multi-section Markdown report          │
└───────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 5: Deep Interaction ───────────────────────┐
│  Chat with ReportAgent (multi-turn + tool calls)  │
│  Interview individual agents in the simulation    │
└───────────────────────────────────────────────────┘
```

### API Endpoints

| Group | Endpoint | Method | Description |
|-------|----------|--------|-------------|
| **Root** | `/` | GET | Service metadata and endpoint discovery |
| **Health** | `/health` | GET | Health check with uptime, config status, active simulations |
| **Graph** | `/api/graph/ontology/generate` | POST | Upload files + generate ontology |
| | `/api/graph/build` | POST | Build Zep knowledge graph |
| | `/api/graph/task/<task_id>` | GET | Query async task status |
| | `/api/graph/data/<graph_id>` | GET | Get graph nodes and edges |
| | `/api/graph/project/list` | GET | List all projects |
| **Simulation** | `/api/simulation/create` | POST | Create simulation |
| | `/api/simulation/prepare` | POST | Generate agent profiles and config |
| | `/api/simulation/start` | POST | Start dual-platform simulation |
| | `/api/simulation/start/status` | POST | Poll simulation run status |
| | `/api/simulation/<id>/results` | GET | Get simulation results |
| | `/api/simulation/interview` | POST | Chat with a simulated agent |
| **Report** | `/api/report/generate` | POST | Generate prediction report |
| | `/api/report/generate/status` | POST | Poll report generation progress |
| | `/api/report/<id>` | GET | Get report details |
| | `/api/report/chat` | POST | Chat with ReportAgent |
| | `/api/report/<id>/download` | GET | Download report as Markdown |

### Persistence Model

MiroFish uses **file-based persistence** (no database required):

- **Projects**: `backend/uploads/projects/<project_id>/project.json` + extracted text + uploaded files
- **Simulations**: `backend/uploads/simulations/<sim_id>/state.json` + agent profiles + config + action logs
- **Reports**: `backend/uploads/reports/<report_id>/report.json` + section Markdown files + agent logs
- **Tasks**: In-memory singleton (lost on restart; async task tracking only)

## Quick Start

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 18+ | `node -v` |
| Python | 3.11 - 3.12 | `python --version` |
| uv | Latest | `uv --version` |

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (see Configuration section below)
```

### 2. Install Dependencies

```bash
npm run setup:all
```

Or step by step:

```bash
npm run setup           # Node dependencies (root + frontend)
npm run setup:backend   # Python dependencies (creates venv via uv)
```

### 3. Start

```bash
npm run dev    # Starts both frontend (port 3000) and backend (port 5001)
```

Individual services:

```bash
npm run backend    # Backend only
npm run frontend   # Frontend only
```

### Docker Deployment

```bash
cp .env.example .env
# Edit .env with your API keys
docker compose up -d
```

Exposes ports 3000 (frontend) and 5001 (backend). Upload data persists in `./backend/uploads/`.

### Production Deployment

The Dockerfile uses a multi-stage build optimized for production:

- **Stage 1**: Builds frontend static assets with Node 18
- **Stage 2**: Runs backend with gunicorn (2 workers, 120s timeout)
- Uses `tini` as PID 1 for proper signal forwarding to simulation subprocesses
- Runs as non-root `mirofish` user
- Includes Docker HEALTHCHECK against `/health` endpoint

```bash
docker build -t mirofish:latest .
docker run -d --name mirofish \
  --env-file .env \
  -p 5001:5001 \
  -v ./backend/uploads:/app/backend/uploads \
  mirofish:latest
```

**Note**: In production, serve the frontend static assets (`frontend/dist/`) via a reverse proxy (nginx/Caddy) pointed at the backend on port 5001. Set `SECRET_KEY` and `CORS_ORIGINS` explicitly.

## Configuration

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for any OpenAI-compatible LLM provider | `sk-...` |
| `LLM_BASE_URL` | Base URL of the LLM API | `https://api.openai.com/v1` |
| `LLM_MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `ZEP_API_KEY` | Zep Cloud API key ([sign up](https://app.getzep.com/)) | `z_...` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | Random per-start | Flask session key (set for persistence across restarts) |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `FLASK_DEBUG` | `True` | Enable debug mode (disable in production) |
| `VITE_API_BASE_URL` | (empty, uses proxy) | Frontend API base URL for remote deployments |
| `MIROFISH_FRONTEND_PORT` | `3000` | Vite dev server port |
| `MIROFISH_BACKEND_URL` | `http://localhost:5001` | Vite proxy target |
| `REPORT_AGENT_MAX_TOOL_CALLS` | `5` | Max tool calls per report section |
| `REPORT_TOOL_TIMEOUT` | `120` | Timeout (seconds) for each report tool call |
| `OASIS_DEFAULT_MAX_ROUNDS` | `10` | Default simulation rounds |

### Supported LLM Providers

MiroFish works with any OpenAI SDK-compatible API:

| Provider | `LLM_BASE_URL` | Recommended Model |
|----------|----------------|-------------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-70b-versatile` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Alibaba Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |

> **Note**: Providers that don't support `response_format={"type": "json_object"}` are handled automatically via fallback.

## Security

- **SECRET_KEY**: Auto-generated if not set; logs a warning. Set explicitly in production.
- **CORS**: Restricted to configured origins (not wildcard) by default.
- **Traceback leaks**: Stack traces are only returned in API responses when `FLASK_DEBUG=True`.
- **Input validation**: `simulation_requirement` is length-capped to prevent prompt injection.
- **File uploads**: Limited to 50MB; allowed extensions: PDF, MD, TXT, Markdown.
- **No authentication**: MiroFish does not implement user authentication. Deploy behind a reverse proxy with auth for production use.

## Observability

- **Structured logging**: All log entries include a correlation ID for request tracing.
- **Correlation ID**: Auto-generated per request, or pass `X-Correlation-ID` header.
- **Log files**: Rotated daily in `backend/logs/`, 10MB max per file, 5 backups.
- **Health endpoint**: `/health` returns uptime, config status, active simulation count.
- **Report agent logs**: Each report generates `agent_log.jsonl` with full ReACT execution trace.

## Known Limitations

- **Persistence**: Task state is in-memory only (lost on server restart). Project/simulation/report data persists to disk.
- **No authentication**: Must be deployed behind an auth proxy for multi-user scenarios.
- **Memory usage**: Large simulations (100+ agents, 40+ rounds) consume significant memory due to OASIS agent processes.
- **Zep dependency**: Requires Zep Cloud for graph operations; no offline/local graph alternative.
- **Single-instance**: No horizontal scaling; one backend process handles all simulations.

## Acknowledgments

MiroFish's simulation engine is powered by **[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis)** from the CAMEL-AI team.

## License

[AGPL-3.0](LICENSE)
