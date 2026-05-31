"""
app/main.py  —  FastAPI application entry point

Run with:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.routes import predict, chat, pipeline as pipeline_routes

# ---------------------------------------------------------------------------
# Scheduler  (daily pipeline at midnight Tunis time)
# ---------------------------------------------------------------------------
_scheduler = BackgroundScheduler(timezone="Africa/Tunis")


def _scheduled_pipeline():
    from pipeline import run_pipeline
    run_pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _scheduler.add_job(
        _scheduled_pipeline,
        CronTrigger(hour=0, minute=0),
        id="daily_pipeline",
        replace_existing=True,
    )
    _scheduler.start()
    yield
    _scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Project Risk AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(pipeline_routes.router)


@app.get("/")
def root():
    return {"message": "Project Risk AI API — running"}
