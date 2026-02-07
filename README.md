# MDIMG — Multi-Agent Medical Imaging Quality Assurance

Detect low-quality medical scans (DICOM), suggest corrective measures,
apply AI-based enhancement, and validate against clinical standards —
with full traceability, a web UI, and reproducible builds.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  CLI (main.py)           Flask API (backend/)   React (frontend/) │
│  --input --genai         JSON-only REST API     Vite + React SPA  │
│  --model --max-iters     /api/upload, /api/run  Upload, Runs,     │
│                          /api/runs, /api/chat   Detail + Chat     │
└────────┬─────────────────────┬────────────────────────────────────┘
         │                     │
         └─────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  pipeline/runner.py  │  ← unified entry point
        └──────────┬───────────┘
                   ▼
 ┌───────────────────────────────────────────────────────┐
 │  pipeline/ package                                    │
 │  core_agents.py   — Detection, Recommendation,        │
 │                    Enhancement, Validation, Report    │
 │  genai_agents.py  — Planner, Tuning, Explainability   │
 │  metrics.py       — 16 quality metrics + scoring      │
 │  enhancement.py   — 7-step enhancement pipeline       │
 │  storage.py       — SQLite persistence                │
 │  chat.py          — Agentic chat assistant            │
 │  dicom_io.py      — DICOM loading + report builder    │
 └───────────────────────────────────────────────────────┘
```

### Key Design Rules

- **No PHI / pixels sent to LLM** — only numeric metrics + issue labels + non-PHI
  metadata (Modality, BodyPartExamined, StudyDescription).
- **Deterministic core preserved** — all image processing stays local.
- **Graceful fallback** — if GenAI fails, the pipeline falls back to the
  deterministic `RecommendationAgent` path automatically.
- **Parameter safety** — all LLM-suggested parameters are clamped to safe bounds
  before execution (see `PARAM_BOUNDS` in `pipeline/schemas.py`).
- **Cost guard** — hard-capped LLM calls per run (default 20, configurable via
  `MDIMG_MAX_LLM_CALLS`).

## Quick Start (5-Minute Demo)

### 1. Install Backend

```powershell
# Windows PowerShell
cd mdimg
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux/macOS
cd mdimg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```powershell
# Windows PowerShell — create .env in project root
"OPENAI_API_KEY=sk-your-key-here" | Out-File -FilePath .env -Encoding utf8
"OPENAI_MODEL=gpt-5-mini" | Add-Content -Path .env -Encoding utf8
"MAX_ITERS=2" | Add-Content -Path .env -Encoding utf8
```

```bash
# Linux/macOS — create .env in project root
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5-mini
MAX_ITERS=2
EOF
```

Or manually create `.env` file with those 3 lines.

### 3. Start Backend (Flask API)

```powershell
# Windows PowerShell
python -m backend.app
# → JSON API at http://localhost:5000
```

```bash
# Linux/macOS
python -m backend.app
# → JSON API at http://localhost:5000
```

### 4. Start Frontend (React)

**Note:** Requires Node.js 18+ (download from [nodejs.org](https://nodejs.org))

```powershell
# Windows PowerShell
cd frontend
npm install
npm run dev
# → Vite dev server at http://localhost:5173
```

```bash
# Linux/macOS
cd frontend && npm install && npm run dev
# → Vite dev server at http://localhost:5173
```

### 5. Demo Steps

1. Open **http://localhost:5173**
2. Drop a DICOM file (`.dcm`) onto the upload zone
3. Optionally toggle **GenAI mode** and pick a model
4. Click **Upload & Run**
5. Watch the progress bar poll for completion
6. Explore 8 tabs: **Overview**, **Metrics**, **Plan JSON**, **Validation**, **Visuals**, **Report**, **Logs**, **Chat**
7. Ask a question in the **Chat** tab

### CLI (unchanged)

```powershell
# Deterministic mode
python main.py --input data/sample.dcm --no-show

# GenAI mode
python main.py --input data/sample.dcm --genai --model gpt-5-mini --max-iters 2 --no-show
```

---

**Note on `app.py` vs `backend/app.py`:**
- `app.py` (root) — Legacy Flask+Jinja UI (still works, uses templates/)
- `backend/app.py` — **New JSON-only API** for the React frontend

Use `python -m backend.app` for the new API or `python app.py` for the legacy UI.

## CLI Reference

| Flag           | Default         | Description |
|----------------|-----------------|-------------|
| `--input`      | *(required)*    | Path to a single DICOM file |
| `--output`     | `outputs`       | Output directory for report + visuals |
| `--no-show`    | `false`         | Suppress matplotlib GUI window |
| `--genai`      | `false`         | Enable GenAI agentic mode |
| `--model`      | `gpt-5-mini`    | OpenAI model name |
| `--max-iters`  | `4`             | Max tuning iterations |
| `--plan-only`  | `false`         | Print plan JSON without executing |
| `--no-redact`  | `false`         | Disable metadata redaction |
| `--verbose`    | `false`         | Enable debug logging |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for GenAI)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-mini` | Default model for GenAI agents |
| `MAX_ITERS` | `2` | Max tuning iterations (backend default) |
| `UPLOAD_DIR` | `uploads/` | Upload directory |
| `OUTPUT_DIR` | `outputs/` | Output for reports + images |
| `MDIMG_DB_PATH` | `data/mdimg.db` | SQLite database path |
| `SECRET_KEY` | *(auto-generated)* | Flask secret key |
| `FLASK_DEBUG` | `0` | Set `1` for debug mode |
| `OPENAI_TEMPERATURE` | `0.2` | LLM temperature |
| `OPENAI_MAX_TOKENS` | `4096` | Max completion tokens |
| `MDIMG_MAX_LLM_CALLS` | `20` | Cost guard: max LLM calls per run |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload DICOM (multipart) → `{file_id}` |
| `POST` | `/api/run` | Start pipeline `{file_id, genai?, model?, max_iters?}` → `{run_id}` |
| `GET` | `/api/runs` | List all runs |
| `GET` | `/api/runs/<id>` | Full run detail + chat history |
| `GET` | `/api/runs/<id>/status` | Lightweight status poll |
| `GET` | `/api/runs/<id>/report` | Markdown report |
| `GET` | `/api/runs/<id>/before_after` | Before/after PNG |
| `GET` | `/api/runs/<id>/logs` | Agent trace logs |
| `POST` | `/api/runs/<id>/chat` | Chat `{message}` → `{reply}` |

## Quality Metrics (16)

| # | Metric | Key | Description |
|---|---|---|---|
| 1 | Noise σ | `sigma` | Wavelet noise estimate |
| 2 | Laplacian Var | `lap_var` | Sharpness via Laplacian variance |
| 3 | Contrast std | `std` | Global contrast |
| 4 | Clip Low | `pct_low` | % pixels ≤ 0.01 |
| 5 | Clip High | `pct_high` | % pixels ≥ 0.99 |
| 6 | Entropy | `entropy` | Shannon entropy |
| 7 | Edge Density | `edge_density` | Fraction of edge pixels |
| 8 | Gradient Mean | `gradient_mag_mean` | Mean gradient magnitude |
| 9 | Gradient Std | `gradient_mag_std` | Gradient magnitude std |
| 10 | SNR Proxy | `snr_proxy` | mean / sigma |
| 11 | CNR Proxy | `cnr_proxy` | (p95−p05) / sigma |
| 12 | Laplacian Energy | `laplacian_energy` | Mean squared Laplacian |
| 13 | Histogram Spread | `histogram_spread` | IQR of pixel intensities |
| 14 | Local Contrast Std | `local_contrast_std` | Std of local 7×7 patch std-devs |
| 15 | Gradient Strength | `gradient_strength` | Mean of top-10% gradients |
| 16 | Gradient Entropy | `gradient_entropy` | Shannon entropy of gradient histogram |

## GenAI Agents

| Agent | Role | Output |
|-------|------|--------|
| **GenAIPlannerAgent** | Generates structured enhancement plan from metrics + issues | `EnhancementPlan` JSON |
| **GenAITuningAgent** | Iteratively refines parameters via tool calls (agentic loop) | Best `EnhancementPlan` JSON |
| **GenAIExplainabilityAgent** | Generates clinician-friendly explanation (8 fields) | `ExplainabilityReport` |
| **ChatAssistant** | Answers natural-language questions about completed runs | Free-text reply |

### Function Tools (exposed to agents)

| Tool | Purpose |
|------|---------|
| `tool_get_metrics` | Compute 13 quality metrics for an image |
| `tool_apply_enhancement` | Execute enhancement plan deterministically |
| `tool_validate` | Compute SSIM / PSNR / NIQE / SNR / CNR validation |
| `tool_score_plan` | Compute scalar objective score for plan comparison |

## Enhancement Safeguards

The `apply_enhancements_from_params()` function includes three automatic safeguards:
1. **Halo check** — re-applies unsharp with halved strength if halo artefacts
   detected
2. **Noise amplification guard** — auto-applies light bilateral denoise if noise
   increased post-enhancement
3. **Over-processing guard** — blends 40% of original back in if SSIM < 0.6

## Output

Each run produces:
- **Markdown report** (`{basename}_report.md`) with:
  - Detected issues, recommendations, applied ops
  - Before/after 13-metric quality table
  - Validation results (SSIM, PSNR, SNR, CNR, NIQE-approx)
  - *(GenAI mode)* Plan JSON, iteration table, model settings,
    explainability section with actionable suggestions & next steps
- **Before/after image** (`{basename}_before_after.png`)
- **SQLite record** in `mdimg_runs.db`

## Testing

```bash
pytest tests/ -v
```

Tests use synthetic numpy arrays — no real DICOM files or API keys required.

Test modules:
- `test_detection.py` — 13-metric computation, issue detection
- `test_pipeline.py` — enhancement path, parameter clamping, deterministic E2E
- `test_schemas.py` — Pydantic schema validation, 10-param bounds
- `test_metrics.py` — expanded metrics, validation, objective score
- `test_storage.py` — SQLite persistence, chat history
- `test_flask.py` — Flask route smoke tests

## Linting

```bash
ruff check .
```

## Security & Privacy

- **No raw images or PHI are sent to the LLM.** Only numeric quality metrics
  and non-PHI DICOM metadata are transmitted.
- PHI-pattern sanitisation on all agent traces before storage.
- DICOM metadata is treated as untrusted text — sanitised before prompt injection.
- LLM-suggested parameters are clamped to safe bounds before execution.
- Strict JSON parsing via Pydantic structured output — rejects non-JSON responses.
- Hard-capped LLM calls per run (configurable cost guard).
- Flask CSRF protection via flask-wtf, 50 MB upload limit, path traversal protection.
- If GenAI fails for any reason, falls back to deterministic path.

## File Structure

```
mdimg/
├── main.py                # CLI entry point (unchanged)
├── app.py                 # Legacy Flask+Jinja UI (still works)
├── backend/               # NEW — Flask JSON-only API
│   ├── __init__.py
│   ├── app.py             # Flask API (create_app factory)
│   ├── config.py          # Centralised env config
│   └── pipeline_runner.py # Async wrapper around run_pipeline()
├── frontend/              # NEW — React + Vite SPA
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── App.tsx         # React Router setup
│       ├── main.tsx        # Entry point
│       ├── index.css       # Dark theme
│       ├── api/client.ts   # API client
│       ├── components/     # FileUpload, MetricsTable, ChatPanel, etc.
│       └── pages/          # UploadPage, RunDetailPage, RunsListPage
├── pipeline/              # Core package (shared by CLI + backend)
│   ├── runner.py          # Unified pipeline entry point
│   ├── schemas.py         # Pydantic models (10 tunable params)
│   ├── metrics.py         # 16 quality metrics + validation + scoring
│   ├── enhancement.py     # 7-step enhancement + safeguards
│   ├── dicom_io.py        # DICOM loading, visuals, report builder
│   ├── core_agents.py     # 5 deterministic agents
│   ├── genai_agents.py    # GenAI orchestration (Agents SDK)
│   ├── tools.py           # Function tools for LLM agents
│   ├── chat.py            # Agentic chat assistant
│   ├── storage.py         # SQLite DAO (runs + chat)
│   └── agent_logger.py    # PHI-safe trace logging
├── tests/                 # pytest test suite
├── requirements.txt
├── LICENSE
└── README.md
```

## License

[MIT](LICENSE)
