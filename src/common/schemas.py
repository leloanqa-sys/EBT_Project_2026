from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# =========================================================================
# LAYER 1: QUERY
# =========================================================================

class QueryType(str, Enum):
    KIS = "KIS"
    QA = "QA"
    TRAKE = "TRAKE"

@dataclass
class Query:
    """Raw input query from contest/user benchmark."""
    query_id: str
    raw_text: str
    query_type: QueryType = QueryType.KIS
    question_text: Optional[str] = None


# =========================================================================
# LAYER 2: SEMANTIC (VISUAL IR)
# =========================================================================

class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

class Entity(BaseModel):
    id: str
    label: str

class Attribute(BaseModel):
    entity_id: str
    name: str
    value: Any
    polarity: Polarity = Polarity.POSITIVE

class Relation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    surface_form: Optional[str] = None
    polarity: Polarity = Polarity.POSITIVE

class Event(BaseModel):
    id: str
    action: str
    participants: List[str]
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    polarity: Polarity = Polarity.POSITIVE

class TemporalConstraint(BaseModel):
    source_id: str = ""
    target_id: str = ""
    relation: str = "before"
    event_before: Optional[str] = None
    event_after: Optional[str] = None
    event_id: Optional[str] = None
    target_event_id: Optional[str] = None

class OrderConstraint(BaseModel):
    target_id: str = ""
    axis: str = "time"
    direction: str = "ascending"
    before: Optional[str] = None
    after: Optional[str] = None

class SelectionConstraint(BaseModel):
    target_id: str = ""
    top_k: int = 100
    diversity: bool = True


class ObjectConstraint(BaseModel):
    class_name: str
    min_count: int = 1
    spatial_condition: Optional[str] = None

class MetaInfo(BaseModel):

    parser_confidence: float = 1.0
    ambiguity_notes: str = ""

class VisualIRGraph(BaseModel):
    """
    Role B Contract Output: Visual Intermediate Representation.
    Strictly represents the semantic intent of a query, independent of execution logic.
    """
    ir_version: str = "2.0"
    query_id: str
    raw_text: str
    clip_query_en: str = ""
    relaxed_query_en: str = ""
    query_type: str
    entities: List[Entity] = Field(default_factory=list)
    relaxed_associations: List[Entity] = Field(default_factory=list)
    attributes: List[Attribute] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    temporal_constraints: List[TemporalConstraint] = Field(default_factory=list)
    order_constraints: List[OrderConstraint] = Field(default_factory=list)
    selection_constraints: List[SelectionConstraint] = Field(default_factory=list)
    meta: MetaInfo = Field(default_factory=MetaInfo)


# =========================================================================
# LAYER 2.5: PLANNER & SCORING CONFIG
# =========================================================================

class ScoringPlan(BaseModel):
    """
    Deterministic Planner Output: Configuration for ranking weights and retrieval limits.
    Separated from the semantic parser.

    Weight channels:
      w_visual   — SigLIP cosine similarity (primary signal)
      w_metadata — TITLE + DESCRIPTION score (video-level, penalized ×0.4 in TextRetriever)
      w_ocr      — OCR + ASR score (frame-level, temporally localized). Routed separately
                   from w_metadata to allow Conditional Routing in Exp D.
      w_object   — LOCKED at 0.0 until competition. Do not change.
    """
    context_type: str = Field(default="default")

    # Core weights
    w_visual: float = Field(default=1.0, ge=0.0, le=2.0)
    w_object: float = Field(default=0.0, ge=0.0, le=2.0,
                            description="LOCKED=0.0 for competition. Do not enable.")
    object_min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    w_spatial: float = Field(default=0.0, ge=0.0, le=2.0)
    w_metadata: float = Field(default=0.3, ge=0.0, le=2.0,
                              description="TITLE/DESCRIPTION weight (video-level signal)")
    w_ocr: float = Field(default=0.2, ge=0.0, le=2.0,
                         description="OCR/ASR weight (frame-level signal). "
                                     "Routed separately for Conditional Routing (Exp D).")
    w_temporal: float = Field(default=0.0, ge=0.0, le=2.0)

    # Retrieval and cutoff boundaries
    visual_top_k: int = Field(default=500, ge=20, le=3000)
    candidate_top_k: int = Field(default=100, ge=10, le=500)

    # VLM Escalation
    vlm_required: bool = False
    vlm_top_k: int = Field(default=10, ge=1, le=50)
    rationale: str = ""


# =========================================================================
# LAYER 3: EVIDENCE & PROVENANCE (Decoupled CandidateFrame)
# =========================================================================

@dataclass
class CandidateFrame:
    """
    Base identity of a retrieved frame.
    No scores, no VQA answers, strictly identity and temporal metadata.
    """
    faiss_id: int
    video_id: str
    frame_idx: int
    pts_time: float
    fps: float

@dataclass
class VisualEvidence:
    siglip_score: float = 0.0
    obj_score: float = 0.0
    spatial_score: float = 0.0

@dataclass
class MetadataEvidence:
    score: float = 0.0
    matched_fields: List[str] = field(default_factory=list)

@dataclass
class TemporalWindow:
    start_time: float
    end_time: float

@dataclass
class TemporalEvidence:
    candidate_time: float = 0.0
    window: Optional[TemporalWindow] = None
    overlap: float = 0.0
    score: float = 0.0

@dataclass
class EventCandidate:
    """Useful for TRAKE tracking of temporal sequences"""
    event_id: str
    video_id: str
    start_frame: int
    end_frame: int
    best_frame: int
    confidence: float


# =========================================================================
# LAYER 4: SCORING & RANKING
# =========================================================================

@dataclass
class CandidateScore:
    visual: float = 0.0
    object: float = 0.0
    spatial: float = 0.0
    metadata: float = 0.0
    temporal: float = 0.0
    final: float = 0.0

@dataclass
class RankedCandidate:
    """
    Comprehensive output representing a fully evaluated candidate.
    Contains full provenance for debugging why a frame was ranked at its position.
    """
    candidate: CandidateFrame
    visual_evidence: VisualEvidence
    metadata_evidence: MetadataEvidence
    temporal_evidence: TemporalEvidence
    score: CandidateScore
    rank: int = -1
    vqa_answer: Optional[str] = None
    is_ambiguous: bool = False


# =========================================================================
# LAYER 5: OUTPUT SUBMISSION
# =========================================================================

@dataclass
class Answer:
    answer_text: str
    confidence: Optional[float] = None

@dataclass
class SubmissionItem:
    rank: int
    video_id: str
    frame_id: int
    answer: Optional[str] = None
    vqa_answer: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class SubmissionOutput:
    query_id: str
    query_type: QueryType
    items: List[SubmissionItem] = field(default_factory=list)


# =========================================================================
# LAYER 6: QUERY IR (Merged from nlp.schemas)
# =========================================================================

class VisualTarget(BaseModel):
    """A concept that should be visually present in the frame."""
    concept: str = Field(..., description="English concept/noun for SigLIP query")
    weight: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence this concept is visually salient")


class TextTarget(BaseModel):
    """
    A keyword/phrase that may appear in OCR, ASR transcript, or metadata.
    These are PROBABILISTIC signals — not exact match filters.
    weight reflects confidence that the term appears in text form.
    """
    term: str = Field(..., description="Keyword to search in FTS5 (English preferred)")
    source_priority: List[str] = Field(
        default_factory=lambda: ["ASR", "OCR", "TITLE", "DESCRIPTION"],
        description="Preferred text sources to check, in order"
    )
    weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "0.0 = pure world knowledge inference (no text verification possible), "
            "0.5 = might appear in ASR/metadata, "
            "1.0 = very likely to appear as literal text (e.g. location names shown on screen)"
        )
    )



class QueryIR(BaseModel):
    """
    Structured Intermediate Representation of a user query.
    This is the ONLY output format Gemini is allowed to produce.
    Downstream retrieval modules consume this directly.
    """
    query_id: str
    raw_text: str
    query_type: str = "KIS"

    # ── Query classification (Phase 4) ─────────────────────────────────────
    query_class: str = Field(
        default="Q_COMPOSITE",
        description=(
            "Coarse classification for conditional routing. One of:\n"
            "  Q_VISUAL    — primarily visual (scene, action, appearance)\n"
            "  Q_ENTITY    — named entity (person, place, org, product)\n"
            "  Q_METADATA  — metadata-heavy (title, credits, OCR text)\n"
            "  Q_COMPOSITE — mix of visual + entity/metadata (default)"
        )
    )

    # ── Modality-specific targets ──────────────────────────────────────────
    dense_caption_en: str = Field(
        default="",
        description="A rich, fluent English translation/caption of the visual scene, capturing all colors, actions, and objects."
    )
    visual_targets: List[VisualTarget] = Field(
        default_factory=list,
        description="Independent visual elements to force SigLIP intersection"
    )
    text_targets: List[TextTarget] = Field(
        default_factory=list,
        description="Text strings likely to be found via OCR/ASR"
    )
    object_targets: List[str] = Field(
        default_factory=list,
        description="Pure physical nouns for Bounding Box detection"
    )
    object_constraints: List[ObjectConstraint] = Field(default_factory=list, description="EXPERIMENTAL: Strict counting and spatial constraints")
    temporal_sequence: List[str] = Field(
        default_factory=list,
        description="Ordered list of events for TRAKE (earliest first)"
    )

    # ── Parser confidence & metadata ─────────────────────────────────────────
    parser_mode: str = Field(
        default="GEMINI",
        description="GEMINI or FALLBACK"
    )
    parser_confidence: float = Field(
        default=1.0,
        description="Confidence in parsing quality (0.0-1.0)"
    )
    ambiguity_note: str = Field(
        default="",
        description="Any ambiguity or uncertainty noted during parsing"
    )

    def get_siglip_query(self) -> str:
        """
        Return the query for SigLIP.
        Unlocked: Now prioritizes the rich English translation (dense_caption_en) 
        generated by the LLM, falling back to raw_text if missing.
        """
        if self.dense_caption_en and self.dense_caption_en.strip():
            return self.dense_caption_en.strip()
        return self.raw_text

    def get_multi_target_queries(self) -> List[Tuple[str, float]]:
        """
        Returns multiple visual targets for independent FAISS intersection.
        """
        if self.visual_targets:
            return [(vt.concept, vt.weight) for vt in self.visual_targets]
        if self.dense_caption_en:
            return [(self.dense_caption_en, 1.0)]
        return []

    def get_fts5_query(self) -> Optional[str]:
        """
        Build a probabilistic FTS5 MATCH expression.
        Terms are joined by OR (union semantics) to avoid exact-match brittleness.
        Each term is stemmed by porter tokenizer in FTS5.
        Terms with weight < FTS_MIN_WEIGHT are dropped (pure world-knowledge inference,
        impossible to verify via text; e.g. 'Jaws 1975' from 'Steven Spielberg').
        """
        from src.common.config import FTS_MIN_WEIGHT
        fts_terms = []
        for t in self.text_targets:
            if t.weight >= FTS_MIN_WEIGHT:
                term = t.term.strip()
                if not term:
                    continue
                # Implement Prefix Wildcard to handle Typo/ASR errors
                # Example: 'mazut' -> 'mazu*'
                # Example: 'mazut' -> 'mazu*'
                if " " not in term:
                    if len(term) > 4:
                        term = term[:-1] + "*"
                    else:
                        term = term + "*"
                else:
                    words = term.split()
                    term = " OR ".join(f"{w}*" if len(w) > 4 else w for w in words)
                    term = f"({term})"
                fts_terms.append(term)
        
        if not fts_terms:
            return None
        # FTS5 OR union: finds segments containing ANY of these terms
        return " OR ".join(fts_terms)

    def get_weighted_fts_score_map(self) -> dict:
        """Returns {term: weight} for downstream score blending."""
        from src.common.config import FTS_MIN_WEIGHT
        return {t.term: t.weight for t in self.text_targets if t.weight >= FTS_MIN_WEIGHT}