from src.common.schemas import Query, VisualIRGraph
from src.role_b_nlp.text_normalizer import normalize_text
from src.role_b_nlp.gemini_nlp_engine import compile_to_visual_ir

def parse_query(query: Query) -> VisualIRGraph:
    """
    Compiles raw Query into VisualIRGraph semantic representation.
    """
    norm_text = normalize_text(query.raw_text)
    return compile_to_visual_ir(query.query_id, norm_text, query.query_type)
