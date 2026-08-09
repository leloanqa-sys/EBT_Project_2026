import re
from typing import List
from src.common.schemas import Query, ParsedQuery, QueryType
from src.role_b_nlp.text_normalizer import normalize_text
from src.role_b_nlp.question_classifier import classify_question
from src.role_b_nlp.gemini_nlp_engine import extract_target_objects, decompose_events

def extract_sub_events(text: str) -> List[str]:
    """
    Tách các sự kiện con cho TRAKE dưới dạng List[str] thực dụng (Regex fallback).
    """
    if not text:
        return []

    # Pattern 1: (1) ... (2) ... hoặc 1. ... 2. ... hoặc bước 1 / step 1
    pattern_numbered = re.split(r'(?:\(\d+\)|\b\d+[\.\)]|\b(?:bước|step)\s+\d+:?)', text, flags=re.IGNORECASE)
    parts = [normalize_text(p) for p in pattern_numbered if normalize_text(p)]
    
    if len(parts) > 1:
        return parts

    # Pattern 2: Từ nối thời gian Tiếng Việt + Tiếng Anh ("sau đó", "tiếp theo", "then", "after that", "next")
    pattern_transitions = re.split(r'\b(?:sau đó|tiếp theo|rồi|cuối cùng|then|after that|next|followed by|finally)\b', text, flags=re.IGNORECASE)
    parts_trans = [normalize_text(p) for p in pattern_transitions if normalize_text(p)]
    
    if len(parts_trans) > 1:
        return parts_trans

    # Fallback: Trả về câu đơn
    return [normalize_text(text)]

def parse_query(query: Query) -> ParsedQuery:
    """
    Biến đổi raw Query thành ParsedQuery thực dụng theo schema mới.
    Sử dụng Gemini NLP Engine với cơ chế fallback 2 tầng.
    """
    norm_text = normalize_text(query.raw_text)
    sub_events = []
    question_type = None
    qa_prompt = None
    attributes = {}

    target_objs = extract_target_objects(query.raw_text)
    if target_objs:
        attributes["objects"] = {"keywords": target_objs}

    if query.query_type == QueryType.TRAKE:
        try:
            sub_events = decompose_events(query.raw_text)
            if not sub_events:
                raise ValueError("Gemini returned empty sub_events")
        except Exception:
            sub_events = extract_sub_events(query.raw_text)
            if not sub_events:
                sub_events = [norm_text]
    elif query.query_type == QueryType.QA:
        question_str = query.question_text if query.question_text else query.raw_text
        question_type = classify_question(question_str)
        qa_prompt = f"Question: {question_str} Answer:"

    return ParsedQuery(
        normalized_text=norm_text,
        query_type=query.query_type,
        sub_events=sub_events,
        question_type=question_type,
        qa_prompt=qa_prompt,
        attributes=attributes
    )

