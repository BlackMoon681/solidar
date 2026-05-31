from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/pipeline/status")
def pipeline_status():
    from pipeline import get_status
    return get_status()


@router.post("/pipeline/run")
def pipeline_run(
    background_tasks: BackgroundTasks,
    force:  bool = Query(False, description="Skip change-detection and run all stages"),
    stages: str  = Query(None,  description="Comma-separated stages: etl,chatbot,model,figures"),
):
    stage_list = [s.strip() for s in stages.split(",")] if stages else None

    def _run():
        from pipeline import run_pipeline
        run_pipeline(stages=stage_list, force=force)

    background_tasks.add_task(_run)
    return {
        "status": "triggered",
        "force":  force,
        "stages": stage_list or ["etl", "chatbot", "model", "figures"],
        "note":   "Pipeline running in background. Poll /pipeline/status for updates.",
    }


@router.get("/model/versions")
def model_versions():
    from ml.versioning import get_all, current_promoted_version
    versions    = get_all()
    current_ver = current_promoted_version()
    return {"total": len(versions), "current": current_ver, "versions": versions}


@router.post("/model/rollback/{version}")
def model_rollback(version: int):
    from ml.versioning import get_all, promote
    versions = get_all()
    target   = next((e for e in versions if e["version"] == version), None)
    if not target:
        return JSONResponse(status_code=404, content={"error": f"Version {version} not found"})
    promote(version)
    return {
        "status":  "rolled_back",
        "version": version,
        "tag":     target["tag"],
        "auc":     target["holdout_auc"],
        "note":    "Restart the API to load the restored model.",
    }
