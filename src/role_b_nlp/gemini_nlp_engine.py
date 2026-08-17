import os
import json
import hashlib
import requests
import time
import threading
from typing import List, Optional, Any, Dict

from src.common.schemas import (
    VisualIRGraph, Entity, Attribute, Relation, Event,
    TemporalConstraint, OrderConstraint, SelectionConstraint, Polarity
)

CACHE_DIR = os.path.join("data", "cache", "gemini_nlp")
os.makedirs(CACHE_DIR, exist_ok=True)

def load_gemini_api_key() -> str:
    """
    Reads Gemini API key from .env, .env.example or environment variables.
    Checks API_GEMINI, GEMINI_API_KEY, GEMINI_KEY, GOOGLE_API_KEY.
    """
    for key_name in ["API_GEMINI", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key_name)
        if val and val.strip():
            return val.strip()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for env_file in [os.path.join(PROJECT_ROOT, ".env"), os.path.join(PROJECT_ROOT, ".env.example")]:
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in ["API_GEMINI", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"] and v:
                            return v
    return ""

def _get_cache_path(query_text: str, task: str) -> str:
    h = hashlib.sha256(f"{task}_{query_text}".encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def _read_cache(query_text: str, task: str):
    p = _get_cache_path(query_text, task)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None

def _write_cache(query_text: str, task: str, data: dict):
    p = _get_cache_path(query_text, task)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

import warnings
import urllib3
warnings.simplefilter('ignore', urllib3.exceptions.InsecureRequestWarning)

def get_available_gemini_models(api_key: Optional[str] = None) -> List[dict[str, Any]]:
    """
    Trích xuất danh sách các models Gemini active và thông số quota (input/output token limits).
    """
    key = api_key or load_gemini_api_key()
    if not key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        res = requests.get(url, timeout=10.0, verify=False)
        if res.status_code == 200:
            models_raw = res.json().get('models', [])
            available = []
            for m in models_raw:
                methods = m.get('supportedGenerationMethods', [])
                if 'generateContent' in methods:
                    clean_name = m.get('name', '').replace("models/", "")
                    available.append({
                        "model_name": clean_name,
                        "display_name": m.get('displayName', ''),
                        "input_token_limit": m.get('inputTokenLimit', 0),
                        "output_token_limit": m.get('outputTokenLimit', 0),
                        "supported_methods": methods
                    })
            return available
    except Exception as e:
        print(f"[NLP-API-ERROR] Error fetching Gemini models list: {e}")
    return []

# Dynamic or active model cascade based on available Gemini models
GEMINI_MODEL_CASCADE = [
    "gemini-3.6-flash",        # Latest active standard model
    "gemini-3.5-flash",        # Stable active fallback
    "gemini-3.1-flash-lite",   # Fast active fallback
]

def _exponential_backoff(attempt: int, base_delay: float = 2.0):
    """Calculates exponential backoff delay: 2s, 4s, 8s..."""
    return base_delay * (2 ** attempt)

_last_request_time = 0.0
_request_lock = threading.Lock()
_current_model_idx = 0

def _get_next_model():
    global _current_model_idx
    with _request_lock:
        model = GEMINI_MODEL_CASCADE[_current_model_idx]
        _current_model_idx = (_current_model_idx + 1) % len(GEMINI_MODEL_CASCADE)
        return model

def compile_to_visual_ir(query_id: str, raw_text: str, query_type: str) -> VisualIRGraph:
    """
    Compiles a natural language query into a VisualIRGraph using Gemini.
    """
    if not raw_text.strip():
        return VisualIRGraph(query_id=query_id, raw_text=raw_text, query_type=query_type)

    cached = _read_cache(raw_text, "visual_ir")
    if cached is not None:
        try:
            return VisualIRGraph(**cached)
        except Exception:
            pass

    api_key = load_gemini_api_key()
    if not api_key:
        # Fallback empty graph if no key
        return VisualIRGraph(query_id=query_id, raw_text=raw_text, query_type=query_type)

    schema_str = """
{
  "clip_query_en": "a girl alone in a red shirt",
  "entities": [{"id": "e1", "label": "person"}],
  "attributes": [{"entity_id": "e1", "name": "shirt_color", "value": "red", "polarity": "POSITIVE"}],
  "relations": [{"source_id": "e1", "target_id": "e2", "relation_type": "behind", "surface_form": "phía sau", "polarity": "POSITIVE"}],
  "events": [{"id": "v1", "action": "argue", "participants": ["e1", "e2"], "polarity": "POSITIVE"}],
  "temporal_constraints": [{"source_id": "v1", "target_id": "v2", "relation": "after"}],
  "order_constraints": [{"target_id": "e1", "axis": "horizontal", "direction": "left_to_right"}],
  "selection_constraints": [{"target_id": "e1", "rank": 2}],
  "meta": {"confidence": 0.9, "ambiguous_notes": ""},
  "scoring_plan": {
    "context_type": "color_attribute",
    "w_clip": 1.5,
    "w_obj": 0.2,
    "w_spatial": 0.3,
    "vlm_required": true,
    "vlm_top_k": 10,
    "clip_k": 500,
    "rationale": "Query contains color detail 'red shirt' which Faster R-CNN cannot verify directly"
  }
}
    """

    prompt = f"""
You are a Computer Vision Semantic Compiler and Query Planner for a video-retrieval competition (AIC 2026). Your job is to convert one natural-language query (Vietnamese or English) into the Visual IR Graph schema below, by reasoning through a fixed set of SKILLS in order.

BEFORE YOU START — QUERY TYPE CHECK:
Identify if the query is: (1) Textual KIS (single scene), (2) Q&A (scene description + factual question), or (3) TRAKE (sequential sub-events).

SKILL 0 — CLIP QUERY TRANSLATION
Provide a smooth, natural English translation of the visual scene in `clip_query_en`. Remove ALL quotation marks, special characters, and punctuation that might break CLIP tokenization.

SKILL 1 — ENTITY GROUNDING
Trigger: any noun phrase referring to physical objects/people. Canonical English names.

SKILL 2 — ATTRIBUTE BINDING
Trigger: colors, clothing, counts, descriptors. Bind explicitly.

SKILL 3 — RELATION EXTRACTION
Trigger: spatial, possessive, wearing, holding, or containment relations.

SKILL 4 — EVENT EXTRACTION
Trigger: action verb phrases (e.g. running, arguing).

SKILL 5 — TEMPORAL SEQUENCING
Trigger: sequential actions, time relation indicators ("before", "after").

SKILL 6 — SPATIAL ORDERING
Trigger: spatial positions (e.g. leftmost, second from right).

SKILL 7 — NEGATION SWEEP
Scope negation over attributes/relations/events with NEGATIVE polarity.

SKILL 8 — SCORING PLAN (DYNAMIC WEIGHT MATRIX)
Based on the query context, define the best execution weights (w_clip, w_obj, w_spatial) to optimize accuracy and prevent object-score bias (e.g., person-detection hubness):
1. context_type: Choose exactly one of: "default", "color_attribute", "scene_context", "spatial_heavy", "action_event", "trake_sequence".
2. w_clip (range: [0.0, 2.0]): Scale higher for scenes (1.8) or color/attributes (1.5) where CLIP has superior understanding.
3. w_obj (range: [0.0, 2.0]): Scale lower for scene_context (0.05) or color_attribute (0.2) to prevent general object tags (e.g. person, clothing) from flooding results. Scale higher (0.6 - 1.0) for concrete physical objects.
4. w_spatial (range: [0.0, 2.0]): Scale higher (1.2) for spatial relations.
5. vlm_required: Set to true if the query contains color descriptors, actions, text, or complex relationships that require visual validation.
6. vlm_top_k (range: [1, 20]): Number of top candidates to escalate (Free key safety: limit to max 10 if vlm_required is true).
7. clip_k (range: [100, 1000]): Retrievable candidates from CLIP (default 500).

SELF-CHECK BEFORE OUTPUT:
- All constraints and references must point to existing entity ids.
- Scoring plan weight ranges are strictly bounded.

OUTPUT CONTRACT:
Return ONLY the JSON object matching this exact schema — no markdown fences, no prose before or after: 
{schema_str}

Query: "{raw_text}"
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    import warnings
    import urllib3
    warnings.simplefilter('ignore', urllib3.exceptions.InsecureRequestWarning)

    max_retries = 3
    for attempt in range(max_retries):
        model_name = _get_next_model()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            # Added verify=False to bypass Windows local SSL issues
            resp = requests.post(url, json=payload, timeout=60.0, verify=False)
            if resp.status_code == 200:
                res_json = resp.json()
                raw_text_resp = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                if raw_text_resp.startswith("```json"):
                    raw_text_resp = raw_text_resp[7:]
                if raw_text_resp.endswith("```"):
                    raw_text_resp = raw_text_resp[:-3]
                
                parsed = json.loads(raw_text_resp)
                # Fill missing arrays
                parsed.setdefault("entities", [])
                parsed.setdefault("attributes", [])
                parsed.setdefault("relations", [])
                parsed.setdefault("events", [])
                parsed.setdefault("temporal_constraints", [])
                parsed.setdefault("order_constraints", [])
                parsed.setdefault("selection_constraints", [])
                
                parsed["query_id"] = query_id
                parsed["raw_text"] = raw_text
                parsed.setdefault("clip_query_en", "")
                parsed["query_type"] = query_type
                parsed["ir_version"] = "1.0"
                
                ir_graph = VisualIRGraph(**parsed)
                _write_cache(raw_text, "visual_ir", parsed)
                return ir_graph
            elif resp.status_code == 429:
                delay = _exponential_backoff(attempt)
                print(f"[NLP-API-WARNING] 429 Too Many Requests. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                print(f"[NLP-API-ERROR] Unexpected Status Code {resp.status_code}: {resp.text}")
                continue
        except requests.exceptions.SSLError as e:
            print(f"[NLP-API-ERROR] SSL Error: {e}")
            break # SSL error won't fix itself on retry
        except requests.exceptions.RequestException as e:
            print(f"[NLP-API-ERROR] Connection Error: {e}")
            delay = _exponential_backoff(attempt)
            time.sleep(delay)
            continue
        except Exception as e:
            print(f"[NLP-API-ERROR] Unexpected Error: {e}")
            continue

    return VisualIRGraph(query_id=query_id, raw_text=raw_text, query_type=query_type)
