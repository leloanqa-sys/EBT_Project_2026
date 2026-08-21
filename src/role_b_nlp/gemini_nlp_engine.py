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
    "gemini-3.5-flash-lite",   # Latest active lite model
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
  "clip_query_en": "a girl in a red shirt wearing a straw conical hat",
  "relaxed_query_en": "a person in bright clothing wearing headwear",
  "entities": [{"id": "e1", "label": "person"}],
  "relaxed_associations": [{"id": "r1", "label": "person"}],
  "attributes": [{"entity_id": "e1", "name": "shirt_color", "value": "red", "polarity": "POSITIVE"}],
  "relations": [{"source_id": "e1", "target_id": "e2", "relation_type": "behind", "surface_form": "phía sau", "polarity": "POSITIVE"}],
  "events": [{"id": "v1", "action": "argue", "participants": ["e1", "e2"], "polarity": "POSITIVE"}],
  "temporal_constraints": [{"source_id": "v1", "target_id": "v2", "relation": "after"}],
  "order_constraints": [{"target_id": "e1", "axis": "horizontal", "direction": "left_to_right"}],
  "selection_constraints": [{"target_id": "e1", "rank": 2}],
  "meta": {"confidence": 0.9, "ambiguous_notes": ""},
  "scoring_plan": {
    "context_type": "color_attribute",
    "w_siglip": 1.5,
    "w_obj": 0.2,
    "w_spatial": 0.3,
    "vlm_required": true,
    "vlm_top_k": 10,
    "siglip_k": 500,
    "rationale": "Query contains color detail 'red shirt' which Faster R-CNN cannot verify directly"
  }
}
    """

    # Load empirical tuning results if available to provide as reference context to Gemini
    tuning_context_str = ""
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tuning_file = os.path.join(_project_root, "outputs", "tuning_results.json")
    if os.path.exists(tuning_file):
        try:
            with open(tuning_file, "r", encoding="utf-8") as f:
                tdata = json.load(f)
                cat_w = tdata.get("category_weights", {})
                glob_w = tdata.get("global_weights", {})
                tuning_context_str = f"\nEMPIRICAL BENCHMARK WEIGHTS (Learned from human verdicts):\n- Global default: {glob_w}\n- Category profiles: {cat_w}\n"
        except Exception:
            pass

    prompt = f"""
You are an expert Vision-Language Compiler and Intelligent Query Planner for the AIC 2026 video retrieval system.
Your job is to convert natural-language queries (Vietnamese or English) into an executable Visual IR Graph schema with DUAL-LAYER RETRIEVAL capabilities (Strict Layer and Relaxed Conceptual Association Layer), and intelligently plan execution weights.

DOWNSTREAM SCORING ARCHITECTURE CONTEXT:
1. Retrieval Backbone: SigLIP2 (768-dim dense semantic embeddings). `norm_siglip` (0.0 to 1.0) captures overall scene semantics, visual concepts, colors, and atmosphere.
2. Object Detector: Faster R-CNN on 80+ COCO classes. `obj_score` (0.0 to 1.0) measures physical presence of concrete objects (car, person, dog, chair). WARNING: It produces noise for generic tags ('person', 'clothing') and CANNOT detect colors or actions!
3. Spatial Engine: Bounding-box geometry. `spatial_score` (0.0 to 1.0) checks positional relations (left_of, right_of, above, below, behind, front).
4. VLM Verifier: Gemini Vision 1.5/2.0. Escalated candidates receive a +5.0 bonus when visual verification confirms complex actions, OCR text, or subtle attributes.
5. Final Ranking Fusion: Fusion_Score = (w_siglip * norm_siglip) + (w_obj * obj_score) + (w_spatial * spatial_score) + VLM_Bonus.
{tuning_context_str}
SKILL 0 — TRACK CLASSIFICATION (query_type):
Determine if the `raw_text` belongs to one of three tracks:
- "TRAKE": If the query explicitly describes a sequence of events over time or steps (e.g. "Phân cảnh 1", "Bước 1", "Step 1", "Sau đó").
- "QA": If the query asks a direct question (e.g. "What is he holding?", "Người đó đang làm gì?", "Trong tay có gì?").
- "KIS": Otherwise, if it's just describing a visual scene or object.

SKILL 1 — TRANSLATION: Produce extremely concise, comma-separated keywords in `clip_query_en`. DO NOT write full descriptive sentences. Extract ONLY the 3-6 most critical visual entities and properties (e.g., "women harvesting pineapples, conical hat, blue boat", "octopus plush toy, girl holding bag"). This is crucial for SigLIP zero-shot matching.
SKILL 1.5 — DUAL-LAYER SEMANTIC RELAXATION:
- Produce `relaxed_query_en`: A semantically associated, broader version of the query (e.g. 'conical straw hat' -> 'headwear / hat', 'red shirt' -> 'bright clothing', 'jumping into pool' -> 'outdoor activity water'). DO NOT drop the core visual theme; generalize concepts into broader categories for fallback retrieval.
- Produce `relaxed_associations`: General entity labels corresponding to relaxed visual categories.
SKILL 2 — ENTITY GROUNDING: Extract physical entities with canonical English names.
SKILL 3 — ATTRIBUTE BINDING: Extract colors, counts, clothing, text.
SKILL 4 — RELATION EXTRACTION: Extract spatial and interaction relations.
SKILL 5 — EVENT/ACTION: Extract action verbs (running, holding, jumping).
SKILL 6 — TEMPORAL / SPATIAL CONSTRAINTS: Extract sequence (before/after) and spatial alignment.
SKILL 7 — SCORING PLAN (INTELLIGENT WEIGHT INTERPOLATION):
Reason about the query characteristics and dynamically set the weights (range 0.0 to 2.0):
- context_type: Choose one of: "color_attribute", "scene_context", "spatial_heavy", "action_event", "object_heavy", "default".
- Reference anchors for your interpolation:
  * "color_attribute" (e.g. 'người áo xanh'): w_siglip=1.5, w_obj=0.2 (lower obj to avoid person-bias), w_spatial=0.3, vlm_required=true (to verify exact color).
  * "scene_context" (e.g. 'bãi biển hoàng hôn'): w_siglip=1.8, w_obj=0.05, w_spatial=0.1, vlm_required=false.
  * "spatial_heavy" (e.g. 'người bên trái ô tô'): w_siglip=0.8, w_obj=0.6, w_spatial=1.2, vlm_required=false.
  * "action_event" (e.g. 'cầm ly nước uống'): w_siglip=1.6, w_obj=0.3, w_spatial=0.2, vlm_required=true (BBox cannot check action).
  * "object_heavy" (e.g. '2 chiếc xe máy và 1 con chó'): w_siglip=1.0, w_obj=0.9, w_spatial=0.4, vlm_required=false.
- Use your reasoning to fine-tune w_siglip, w_obj, w_spatial around these anchors to achieve optimal retrieval accuracy.
- vlm_required: Set true if the query demands visual verification (color, text/OCR, fine action, facial detail).
- vlm_top_k: Limit to max 10 to preserve API quota.
- siglip_k: Candidate retrieval budget (default 500).

OUTPUT CONTRACT:
Return ONLY the JSON object matching this exact schema — no markdown fences, no prose:
{{
  "query_type": "string (KIS, QA, or TRAKE)",
  "clip_query_en": "string",
  "relaxed_query_en": "string",
  "relaxed_associations": [{{"id": "e1", "label": "string"}}],
  "entities": [{{"id": "e1", "label": "string"}}],
  "attributes": [{{"entity_id": "e1", "name": "color", "value": "red", "polarity": "POSITIVE"}}],
  "relations": [{{"source_id": "e1", "target_id": "e2", "relation_type": "left_of", "polarity": "POSITIVE"}}],
  "events": [{{"id": "ev1", "action": "running", "participants": ["e1"], "polarity": "POSITIVE"}}],
  "temporal_constraints": [],
  "order_constraints": [],
  "selection_constraints": [],
  "scoring_plan": {{
    "context_type": "string",
    "w_siglip": 1.0,
    "w_obj": 0.5,
    "w_spatial": 0.5,
    "vlm_required": false,
    "vlm_top_k": 10,
    "siglip_k": 500,
    "rationale": "string"
  }}
}}

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
        model_name = GEMINI_MODEL_CASCADE[min(attempt, len(GEMINI_MODEL_CASCADE) - 1)]
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
                parsed.setdefault("relaxed_associations", [])
                parsed.setdefault("attributes", [])
                parsed.setdefault("relations", [])
                parsed.setdefault("events", [])
                parsed.setdefault("temporal_constraints", [])
                parsed.setdefault("order_constraints", [])
                parsed.setdefault("selection_constraints", [])
                
                # Robust constraint sanitization to prevent LLM schema drift errors
                sanitized_orders = []
                for o in parsed.get("order_constraints", []):
                    if isinstance(o, str):
                        sanitized_orders.append({"target_id": o, "axis": "time", "direction": "ascending"})
                    elif isinstance(o, dict):
                        sanitized_orders.append(o)
                parsed["order_constraints"] = sanitized_orders

                sanitized_temp = []
                for t in parsed.get("temporal_constraints", []):
                    if isinstance(t, dict):
                        source = t.get("source_id") or t.get("event_before") or t.get("event_id") or t.get("before") or ""
                        target = t.get("target_id") or t.get("event_after") or t.get("target_event_id") or t.get("after") or ""
                        rel = t.get("relation") or "before"
                        sanitized_temp.append({
                            "source_id": str(source),
                            "target_id": str(target),
                            "relation": str(rel)
                        })
                parsed["temporal_constraints"] = sanitized_temp
                
                parsed["query_id"] = query_id
                parsed["raw_text"] = raw_text
                parsed.setdefault("clip_query_en", "")
                parsed.setdefault("relaxed_query_en", "")
                if query_type and query_type != "AUTO":
                    parsed["query_type"] = query_type
                else:
                    parsed.setdefault("query_type", "KIS")
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
