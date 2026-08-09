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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# ── Resolve project root (EBT_Project_2026/) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy-init heavy resources on startup ──
_searcher = None

def _get_searcher():
    """Singleton getter — imports VectorSearcher only once."""
    global _searcher
    if _searcher is None:
        try:
            from src.role_a_retrieval.searcher import VectorSearcher
            _searcher = VectorSearcher()
        except Exception as e:
            print(f"[WARN] VectorSearcher unavailable: {e}")
            _searcher = None
    return _searcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load searcher on startup so first request is fast."""
    print("[API] Starting EBT Vision Search API...")
    print(f"[API] Project root: {PROJECT_ROOT}")

    keyframes_dir = PROJECT_ROOT / "data" / "raw" / "keyframes"
    if keyframes_dir.exists():
        print(f"[API] Keyframes directory found: {keyframes_dir}")
    else:
        print(f"[WARN] Keyframes directory NOT found: {keyframes_dir}")

    # Try to pre-load searcher (non-blocking if fails)
    try:
        _get_searcher()
        if _searcher:
            print("[API] VectorSearcher loaded successfully.")
        else:
            print("[API] VectorSearcher not available — running in DEMO mode.")
    except Exception as e:
        print(f"[API] VectorSearcher load error (DEMO mode): {e}")

    yield
    print("[API] Shutting down...")


# ── FastAPI App ──
app = FastAPI(
    title="EBT Vision Search API",
    description="AI Challenge 2026 — Video Retrieval System API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow local dev) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Static Files ──
# Frontend UI
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Keyframe images — served at /keyframes/{video_id}/{frame_idx}.jpg
keyframes_dir = PROJECT_ROOT / "data" / "raw" / "keyframes"
if keyframes_dir.exists():
    app.mount("/keyframes", StaticFiles(directory=str(keyframes_dir)), name="keyframes")

# ── Import & include route modules ──
from api.routes.kis_routes import router as kis_router
app.include_router(kis_router, prefix="/api/v1")


# ── Root redirect → UI ──
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the search UI."""
    return RedirectResponse(url="/static/index.html")


# ── Health check ──
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "searcher_loaded": _searcher is not None,
        "project_root": str(PROJECT_ROOT),
        "keyframes_available": keyframes_dir.exists(),
    }

