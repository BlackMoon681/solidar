# PPP-Risk — AI Platform for Tunisian Public Works Projects

End-to-end platform combining **FastAPI**, **XGBoost**, **FAISS RAG**, and an **automated ML pipeline** to monitor and predict risk in Tunisian public/environmental projects (PPP and PGD).

Built as part of a final engineering thesis at Tek-Up — Wevioo internship.

---

## Architecture

```
app/                   FastAPI application
  main.py              App factory, CORS, APScheduler lifespan
  schemas.py           Pydantic input/output models
  dependencies.py      ML model + FAISS index loaded at startup
  routes/
    predict.py         POST /predict  — risk scoring
    chat.py            POST /chat     — RAG chatbot (multi-turn, history-aware)
    pipeline.py        GET/POST /pipeline/* and /model/*

core/config.py         Centralised DB + data-warehouse config

etl/                   ETL pipeline (extract → transform → load)
  extract.py           Pull raw rows from Joget (jwdb)
  transform.py         Clean facts, resolve FK labels, currency → TND
  load.py              Write dimensions + facts to cleaned_dw warehouse
  main.py              Pipeline orchestrator for ETL stages
  dimensions.py        Dimension-table specs and build logic (galaxy schema)

ml/                    Machine learning
  train.py             XGBoost training (synthetic or real DB data)
  versioning.py        Model version registry + AUC gate + rollback
  report.py            Version comparison report + AUC trend plot

Chatbot/
  data_prep.py         Build documents.jsonl from DB
  build_index.py       Build FAISS index from documents
  data/                documents.jsonl + documents_with_meta.json
  index/               faiss_index.bin

pipeline.py            Nightly orchestrator (ETL + chatbot + ML + figures)

tests/
  test_functional.py   Behavioural tests on ML + FAISS + pipeline
  test_performance.py  Benchmarks (inference, FAISS, feature eng)
  test_security.py     Security checks (CORS, secrets, SQL injection)
  test_db_optimization.py  MySQL index audit + query timing
  run_all.py           Full test suite runner

scripts/
  analystes.py                    Thesis statistics (chapter 4)
  generate_chapter05_figures.py   Chapter 5 figures
  generate_activity_etl.py        ETL activity diagram (galaxy-schema pipeline)
  generate_archi.py               5-layer platform architecture diagram
  generate_dw_schema.py           cleaned_dw galaxy-schema ERD
```

---

## Data Warehouse — Galaxy Schema

The ETL now targets a dedicated MySQL warehouse (`cleaned_dw`) structured as a galaxy schema with **3 fact tables** and **shared dimension tables**:

| Table | Description |
|---|---|
| `fact_ppp` | PPP project risk indicators + resolved label columns |
| `fact_pgd` | PGD project budgets (normalised to TND) |
| `fact_pgd_etude` | PGD study costs (normalised to TND) |
| `dim_axe`, `dim_sous_axe`, … | ~15 dimension tables (UUID-keyed, from Joget param tables) |
| `dim_gouvernorat`, `dim_delegation` | Geographic dimensions (integer-coded) |

The warehouse is created automatically on first run (`etl/load.py:ensure_warehouse`).

---

## Key Metrics

| Component | Value |
|---|---|
| Model AUC-ROC (holdout) | **0.9733** |
| Model AUC-ROC (CV 15-fold) | **0.9746 ± 0.0026** |
| F1-Score (threshold 0.47) | **0.9111** |
| XGBoost inference | **3.5 ms** avg |
| Full prediction pipeline (offline) | **13.8 ms** avg |
| FAISS search (829 vectors × 384 dim) | **0.056 ms** avg |
| Documents indexed (RAG) | **829** (PPP + PGD + Etude) |
| Features | **13** structural + binary risk signals |

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML model | XGBoost + scikit-learn + IsotonicRegression |
| RAG | FAISS + SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | llama-3.3-70b-versatile via Groq API |
| DB | MySQL (via SQLAlchemy + PyMySQL) |
| Data Warehouse | MySQL `cleaned_dw` — galaxy schema |
| Scheduler | APScheduler (daily at midnight, Africa/Tunis) |
| Dashboard | Apache Superset + Chart.js |
| Platform | Joget DX |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` at project root:
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=jwdb
DB_NAME_DW=cleaned_dw
GROQ_API_KEY=your_groq_key
```

---

## Running

```bash
# API
uvicorn app.main:app --reload

# Pipeline (manual run)
python pipeline.py
python pipeline.py --status
python pipeline.py --force

# Train model
python ml/train.py                  # synthetic data
python ml/train.py --source real    # real DB data (needs >= 150 rows)

# Model version report
python ml/report.py
python ml/report.py --plot
python ml/report.py --rollback 2

# Tests
python tests/run_all.py --skip-db
python tests/run_all.py              # requires DB connection

# Build chatbot index
python Chatbot/data_prep.py          # generate documents from DB
python Chatbot/build_index.py        # build FAISS index

# Generate diagrams
python scripts/generate_activity_etl.py   # ETL activity diagram
python scripts/generate_archi.py          # architecture diagram
python scripts/generate_dw_schema.py      # data warehouse ERD
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Risk score (0–1) + level + LLM explanation |
| `POST` | `/chat` | RAG question answering with multi-turn history |
| `GET` | `/pipeline/status` | DB counts, last run, model AUC, log tail |
| `POST` | `/pipeline/run` | Trigger pipeline (`?force=true&stages=etl,chatbot`) |
| `GET` | `/model/versions` | Full ML version registry |
| `POST` | `/model/rollback/{version}` | Restore a previous model to production |
| `GET` | `/` | Health check |

---

## ML Versioning

Every training run archives the model under `ml/model/versions/vXXX_YYYYMMDD_HHMMSS_risk_model.pkl` and logs metrics to `ml/model/model_registry.json`. A new model is only promoted to production if:

```
AUC_new >= AUC_production - 0.005
```

Otherwise the previous version is automatically restored (rollback).
