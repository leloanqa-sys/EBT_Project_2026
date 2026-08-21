import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from src.common.schemas import CandidateFrame
from src.role_a_retrieval.feature_store import l2_normalize

@dataclass
class AuditReport:
    query_text: str
    query_l2_norm: float
    top1_siglip_score: float
    top1_video_id: str
    top1_frame_idx: int
    flag_low_confidence: bool

def audit_vector_l2_norms(vectors: np.ndarray) -> List[float]:
    """Computes L2 norms for a matrix of vectors."""
    if vectors.ndim == 1:
        vectors = np.atleast_2d(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    return [float(n) for n in norms]

def audit_candidate(candidate: CandidateFrame, query_text: str, query_vec: np.ndarray, threshold: float = 0.25) -> AuditReport:
    """Audits a single top-1 candidate against query vector and text."""
    q_norm = float(np.linalg.norm(query_vec))
    siglip_score = candidate.siglip_score
    
    return AuditReport(
        query_text=query_text,
        query_l2_norm=round(q_norm, 4),
        top1_siglip_score=round(siglip_score, 4),
        top1_video_id=candidate.video_id,
        top1_frame_idx=candidate.frame_idx,
        flag_low_confidence=siglip_score < threshold
    )
