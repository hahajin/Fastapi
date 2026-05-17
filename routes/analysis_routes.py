# routes/analysis_routes.py  — FastAPI version of the Flask analysis_bp
import os
import traceback

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.database import AnalysisResult, Project, get_db
from server.user import User

# These imports remain the same — only the web framework changes
from fea_controller.analysis_controller import AnalysisController
from fea_controller.err_process import AnalysisError, ModelLoadError
from fea_controller.model_loader import ModelLoaderController

router = APIRouter(tags=["analysis"])
analysis_controller = AnalysisController()
model_loader = ModelLoaderController()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    project_id: int
    # additional analysis params forwarded verbatim to the controller
    class Config:
        extra = "allow"

class ProjectIdRequest(BaseModel):
    project_id: int

class ReportRequest(BaseModel):
    project_id: int
    report_type: str = "standard"


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/analyze")
def analyze_project(
    data: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project_id = data.project_id
    print(f"[ANALYSIS ROUTE] Starting analysis for project ID: {project_id}")

    if not db.query(Project).filter_by(id=project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")

    # Load model
    try:
        model = model_loader.load_project_data(project_id, db=db, verbose=True)
    except ModelLoadError as e:
        raise HTTPException(status_code=400, detail=f"Model load error: {e.message}")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to load model")

    if not model:
        raise HTTPException(status_code=500, detail="Model initialization failed")

    # Run analysis
    try:
        results = analysis_controller.analyze_project(data.model_dump(), model)
    except AnalysisError as e:
        raise HTTPException(status_code=500, detail={"message": e.message, "debug": e.debug_info})

    # Lock project
    project = db.get(Project, project_id)
    project.is_locked = True
    project.analysis_count += 1
    db.commit()

    print("[ANALYSIS COMPLETE] ✅")
    return {"status": "success", "results": results}


@router.post("/results/getall")
def get_results(
    data: ProjectIdRequest,
    current_user: User = Depends(get_current_user),
):
    project_id = data.project_id
    result = analysis_controller.get_results(project_id)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result)

    file_path = result.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/octet-stream",
    )


@router.post("/results/getjson")
def get_results_json(
    data: ProjectIdRequest,
    current_user: User = Depends(get_current_user),
):
    project_id = data.project_id
    result = analysis_controller.get_results_json(project_id)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result)

    if "data" in result:
        return {
            "status": "success",
            "project_id": project_id,
            "data": result["data"],
            "format": result.get("format", "json"),
            "file_path": result.get("file_path"),
        }

    file_path = result.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="JSON结果文件不存在")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/json",
    )


@router.post("/reports/generate")
def generate_report(
    data: ReportRequest,
    current_user: User = Depends(get_current_user),
):
    result = analysis_controller.generate_report(
        data.project_id, current_user.id, data.report_type
    )
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result)
    return result


@router.get("/reports/download/{report_id}")
def download_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
):
    report_file = analysis_controller.download_report(report_id)
    if not report_file or not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="报告文件不存在")
    return FileResponse(
        path=report_file,
        filename=os.path.basename(report_file),
        media_type="application/pdf",
    )


@router.post("/unlock")
def unlock_project(
    data: ProjectIdRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = db.get(Project, data.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.is_locked = False
    db.query(AnalysisResult).filter_by(project_id=data.project_id).delete()
    db.commit()

    result = analysis_controller.unlock_project(data.project_id)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result)
    return result
