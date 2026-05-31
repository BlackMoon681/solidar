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
    chat.py            POST /chat     — RAG chatbot
    pipeline.py        GET/POST /pipeline/* and /model/*

core/config.py         Centralised DB config

etl/                   ETL pipeline (extract → transform → load)
  extract.py / transform.py / load.py / main.py

ml/                    Machine learning
  train.py             XGBoost training (synthetic or real DB data)
  versioning.py        Model version registry + AUC gate + rollback
  report.py            Version comparison report + AUC trend plot

chatbot/
  data_prep.py         Build documents.jsonl from DB
  build_index.py       Build FAISS index from documents
  data/                documents.jsonl + documents_with_meta.json
  index/               faiss_index.bin

pipeline.py            Nightly orchestrator (ETL + chatbot + ML + figures)

tests/
  test_functional.py   9 behavioural tests on ML + FAISS + pipeline
  test_performance.py  5 benchmarks (inference, FAISS, feature eng)
  test_security.py     7 security checks (CORS, secrets, SQL injection)
  test_db_optimization.py  MySQL index audit + query timing
  run_all.py           Full test suite runner

scripts/
  analystes.py         Thesis statistics (chapter 4)
  generate_chapter05_figures.py  Generate all 7 chapter 5 figures
```

---

## Key metrics

| Component | Value |
|---|---|
| Model AUC-ROC (holdout) | **0.9733** |
| Model AUC-ROC (CV 15-fold) | **0.9746 ± 0.0026** |
| F1-Score (threshold 0.47) | **0.9111** |
| XGBoost inference | **3.5 ms** avg |
| Full prediction pipeline (offline) | **13.8 ms** avg |
| FAISS search (829 vectors × 384 dim) | **0.056 ms** avg |
| Documents indexed (RAG) | **829** (PPP + PGD + Etude) |
| Features | **34** (4 structural + 3 engineered + 6 composite + 21 binary flags) |

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML model | XGBoost + scikit-learn + IsotonicRegression |
| RAG | FAISS + SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | llama-3.3-70b-versatile via Groq API |
| DB | MySQL (via SQLAlchemy + PyMySQL) |
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
python chatbot/data_prep.py          # generate documents from DB
python chatbot/build_index.py        # build FAISS index
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Risk score (0–1) + level + LLM explanation |
| `POST` | `/chat` | RAG question answering with sources |
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
