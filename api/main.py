"""
EBT Project 2026 — API Gateway
================================
FastAPI application serving:
  - RESTful API endpoints for KIS/QA/TRAKE search pipelines
  - Static frontend UI (Visualizer)
  - Keyframe images from data/raw/keyframes/
"""
import os
import sys
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ── Resolve project root (EBT_Project_2026/) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy-init heavy resources on startup ──
_global_state = {}

def get_db():
    return _global_state.get("db")

def get_faiss():
    return _global_state.get("faiss")

def get_detection_store():
    return _global_state.get("detection_store")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load components on startup so first request is fast."""
    print("[API] Starting EBT Vision Search API...")
    print(f"[API] Project root: {PROJECT_ROOT}")

    keyframes_dir = PROJECT_ROOT / "data" / "raw" / "keyframes"
    if keyframes_dir.exists():
        print(f"[API] Keyframes directory found: {keyframes_dir}")
    else:
        print(f"[WARN] Keyframes directory NOT found: {keyframes_dir}")

    # Load heavy resources once
    try:
        from src.database.db_manager import DatabaseManager
        from src.retrieval.vector_index import FAISSIndex
        from src.database.legacy_detection_store import LegacyDetectionStore
        
        _global_state["db"] = DatabaseManager()
        print("[API] DatabaseManager loaded.")
        _global_state["faiss"] = FAISSIndex()
        print("[API] FAISSIndex pre-loaded successfully.")
        _global_state["detection_store"] = LegacyDetectionStore()
        print("[API] LegacyDetectionStore loaded.")
    except Exception as e:
        print(f"[API] Component load error: {e}")

    yield
    print("[API] Shutting down...")
    _global_state.clear()


# ── FastAPI App ──
app = FastAPI(
    title="EBT Vision Search API",
    description="AI Challenge 2026 — Video Retrieval System API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheStaticMiddleware)

# ── CORS (allow local dev) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Static Files ──
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

keyframes_dir = PROJECT_ROOT / "data" / "raw" / "keyframes"
if keyframes_dir.exists():
    app.mount("/keyframes", StaticFiles(directory=str(keyframes_dir)), name="keyframes")

from api.routes.kis_routes import router as kis_router
from api.routes.trake_routes import router as trake_router
from api.routes.feedback_routes import router as feedback_router

app.include_router(kis_router, prefix="/api/v1")
app.include_router(trake_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "searcher_loaded": _global_state.get("faiss") is not None,
        "project_root": str(PROJECT_ROOT),
        "keyframes_available": keyframes_dir.exists(),
    }

