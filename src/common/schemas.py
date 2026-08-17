from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

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

from pydantic import BaseModel, Field

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
    polarity: Polarity = Polarity.POSITIVE

class TemporalConstraint(BaseModel):
    source_id: str
    target_id: str
    relation: str

class OrderConstraint(BaseModel):
    target_id: str
    axis: str
    direction: str

class SelectionConstraint(BaseModel):
    target_id: str
    rank: int

class MetaInfo(BaseModel):
    confidence: float = 1.0
    ambiguous_notes: str = ""

class ScoringPlan(BaseModel):
    """
    Scoring plan created by the Gemini Planner to dynamically shift weights
    based on query context, protecting API quota and preventing object-score bias.
    """
    context_type: str = Field(default="default", description="One of: default, color_attribute, scene_context, spatial_heavy, action_event, trake_sequence")
    w_clip: float = Field(default=1.0, ge=0.0, le=2.0)
    w_obj: float = Field(default=0.5, ge=0.0, le=2.0)
    w_spatial: float = Field(default=0.5, ge=0.0, le=2.0)
    vlm_required: bool = False
    vlm_top_k: int = Field(default=10, ge=1, le=50)
    clip_k: int = Field(default=500, ge=100, le=1000)
    rationale: str = ""

class VisualIRGraph(BaseModel):
    """
    Role B Contract Output: Visual Intermediate Representation.
    Represents semantic intent of a query, independent of execution logic.
    """
    ir_version: str = "1.0"
    query_id: str
    raw_text: str
    clip_query_en: str = ""
    query_type: str
    entities: List[Entity] = Field(default_factory=list)
    attributes: List[Attribute] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    temporal_constraints: List[TemporalConstraint] = Field(default_factory=list)
    order_constraints: List[OrderConstraint] = Field(default_factory=list)
    selection_constraints: List[SelectionConstraint] = Field(default_factory=list)
    meta: MetaInfo = Field(default_factory=MetaInfo)
    scoring_plan: ScoringPlan = Field(default_factory=ScoringPlan)


@dataclass
class CandidateFrame:
    """
    Role A Contract Output: Khung hình ứng viên trích xuất từ CLIP retrieval engine.
    Đã bổ sung các trường điểm số ML cho Soft Scoring (Machine Learning).
    """
    faiss_id: int
    video_id: str
    frame_idx: int
    pts_time: float = 0.0
    fps: float = 0.0
    clip_score: float = 0.0
    
    # ML Soft Scoring Fields
    obj_score: float = 0.0
    spatial_score: float = 0.0
    fusion_score: float = 0.0
    vqa_answer: Optional[str] = None
    
    # VLM Escalation Flags
    is_ambiguous: bool = False

@dataclass
class RankedCandidate:
    """
    Role B Contract Output: Ứng viên sau khi tổng hợp điểm Object Detection & Metadata.
    """
    candidate: CandidateFrame
    obj_score: float = 0.0
    meta_score: float = 0.0
    fusion_score: float = 0.0

@dataclass
class Answer:
    """Kết quả trả lời cho Q&A."""
    answer_text: str
    confidence: Optional[float] = None

@dataclass
class SubmissionItem:
    """1 ứng viên trong danh sách top 100 câu trả lời."""
    rank: int
    video_id: str
    frame_id: int
    answer: Optional[str] = None
    vqa_answer: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class SubmissionOutput:
    """Output 100 câu trả lời xuất file nộp bài cho BTC."""
    query_id: str
    query_type: QueryType
    items: List[SubmissionItem] = field(default_factory=list)

