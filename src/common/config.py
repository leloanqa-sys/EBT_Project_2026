"""
Phase 4 — Centralized Configuration
=====================================
Single source of truth for all weights and retrieval limits.

RULES:
- No magic numbers scattered across scripts.
- All weights are "Safe Candidate Baseline" until frozen by Real GT.
- Frozen flag documents the official freeze status.
- Do NOT tune these values per individual GT query (= overfitting).
"""

from __future__ import annotations
from src.common.schemas import ScoringPlan

# ─────────────────────────────────────────────────────────────────────────────
# P1: Safe Candidate Baseline
# Status: CANDIDATE — not frozen. Will freeze after Real GT from BTC.
# ─────────────────────────────────────────────────────────────────────────────

# Core scoring plan (visual + metadata)
# w_ocr is separate because it routes through TextRetriever, not ScoringPlan.
SAFE_CANDIDATE_BASELINE = ScoringPlan(
    w_visual=1.0,
    w_metadata=0.3,   # covers TITLE + DESCRIPTION (video-level signals)
    w_object=0.0,     # LOCKED — do not touch before competition
    visual_top_k=500,
    candidate_top_k=100,
)

# OCR contribution weight (frame-level, goes into w_metadata channel of HybridSearcher)
# Separate from TITLE/DESC weight to allow conditional routing (Exp D).
BASELINE_W_OCR: float = 0.2   # candidate — pending Real GT

# Minimum text_target weight threshold below which term is dropped from FTS5 query
# (pure world-knowledge inference, not verifiable as text in the video)
FTS_MIN_WEIGHT: float = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval limits
# ─────────────────────────────────────────────────────────────────────────────

VISUAL_TOP_K: int = 500     # FAISS candidates per query
FINAL_TOP_K: int = 100      # candidates returned to benchmark / submission

# ─────────────────────────────────────────────────────────────────────────────
# Experiment labels (for benchmark log consistency)
# ─────────────────────────────────────────────────────────────────────────────

EXP_A = "ExpA_RawSigLIP"
EXP_B = "ExpB_Baseline"
EXP_C = "ExpC_MultiTargetSigLIP"
EXP_D = "ExpD_ConditionalRouting"

# ─────────────────────────────────────────────────────────────────────────────
# Gate thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Exp C must beat Exp B by at least this delta in MRR before Exp D is attempted.
EXP_C_GATE_DELTA: float = 0.01

# ─────────────────────────────────────────────────────────────────────────────
# Prompt freeze marker (read-only; updated manually when prompt is locked)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_PROMPT_FROZEN: bool = False   # Set True when prompt is locked before final benchmark
GEMINI_PROMPT_VERSION = "phase4_v10"
