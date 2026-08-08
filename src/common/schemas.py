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

@dataclass
class ParsedQuery:
    """
    Role B Contract Output: Thực dụng, hỗ trợ trực tiếp TRAKE & QA mà không cần over-engineering.
    """
    normalized_text: str
    query_type: QueryType
    sub_events: List[str] = field(default_factory=list)      # chỉ có ở TRAKE
    question_type: Optional[str] = None                       # chỉ có ở QA
    qa_prompt: Optional[str] = None                            # chỉ có ở QA
    attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # ép kiểu 2 tầng, đủ dùng

@dataclass
class CandidateFrame:
    """
    Dữ liệu ứng viên khung hình trao đổi giữa Role A, B, C.
    """
    video_id: str
    frame_id: int
    faiss_id: int
    clip_score: float = 0.0
    obj_score: float = 0.0
    meta_score: float = 0.0
    fusion_score: float = 0.0        # đổi tên theo góp ý thực dụng
    # --- optional, Tầng 2, chưa triển khai — bật khi cần ---
    ocr_text: Optional[str] = None
    asr_transcript: Optional[str] = None

@dataclass
class Answer:
    """Kết quả trả lời cho Q&A."""
    answer_text: str
    confidence: Optional[float] = None   # sửa theo góp ý bug quan trọng nhất

@dataclass
class SubmissionItem:
    """1 ứng viên trong danh sách top 100 câu trả lời."""
    rank: int
    video_id: str
    frame_id: int
    answer: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class SubmissionOutput:
    """Output 100 câu trả lời xuất file nộp bài cho BTC."""
    query_id: str
    query_type: QueryType
    items: List[SubmissionItem] = field(default_factory=list)
