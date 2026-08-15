import os
import json
import hashlib
import requests
import time
import threading
from typing import List, Optional, Any

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

def get_available_gemini_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
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
  "entities": [{"id": "e1", "label": "person"}],
  "attributes": [{"entity_id": "e1", "name": "shirt_color", "value": "red", "polarity": "POSITIVE"}],
  "relations": [{"source_id": "e1", "target_id": "e2", "relation_type": "behind", "surface_form": "phía sau", "polarity": "POSITIVE"}],
  "events": [{"id": "v1", "action": "argue", "participants": ["e1", "e2"], "polarity": "POSITIVE"}],
  "temporal_constraints": [{"source_id": "v1", "target_id": "v2", "relation": "after"}],
  "order_constraints": [{"target_id": "e1", "axis": "horizontal", "direction": "left_to_right"}],
  "selection_constraints": [{"target_id": "e1", "rank": 2}],
  "meta": {"confidence": 0.9, "ambiguous_notes": ""}
}
    """

    prompt = f"""
You are a Computer Vision Semantic Compiler for a video-retrieval competition (AIC 2026). Your only job is to convert one natural-language query (Vietnamese or English) into the Visual IR Graph schema below, by reasoning through a fixed set of SKILLS in order. You are not answering the query — you are structuring it. Stay strictly within what the query states or clearly implies; the retrieval system downstream (CLIP embeddings, object detector, temporal DAG executor) will handle fuzzy matching, so your job is faithful structuring, not exhaustive labeling.

BEFORE YOU START — QUERY TYPE CHECK:
The query belongs to one of three official formats defined by the organizers: (1) Textual KIS — a single scene description, (2) Q&A — a scene description plus one factual question about it, (3) TRAKE — a description of an ordered sequence of sub-events within one action. Identify silently which type this is; it changes which skills below are likely relevant, but does not change the output schema.

SKILL 1 — ENTITY GROUNDING (always run first)
Trigger: any noun phrase referring to a visible physical object, person, or group in the scene.
Procedure: create one entity per distinct visible thing the query treats as a separate object. Do not split one object into multiple entities, and do not merge two distinct objects into one. Label each entity in canonical English (lowercase, singular, generic — e.g. the general category of the object, not a brand or overly specific term). If you are unsure of the exact category, use the most general accurate term rather than guessing a specific one.

SKILL 2 — ATTRIBUTE BINDING
Trigger: an adjective, color, clothing detail, count, size, or other descriptive property attached to an entity.
Procedure: attach as {{entity_id, name, value, polarity}}. `name` should be a short canonical property key (what kind of property it is), `value` the property itself, both in English regardless of query language. Only bind attributes explicitly stated — do not infer unstated properties (e.g. do not assume gender, age, or emotion unless the query says so).

SKILL 3 — RELATION EXTRACTION (spatial/possessive)
Trigger: a phrase describing how two entities relate in space or possession within a single instant (position, holding, wearing, containment).
Procedure: pick the relation_type that most literally matches the query's spatial/possessive meaning — prefer the most generic accurate term over a narrow invented one. Always keep the original phrase in surface_form so downstream logging can catch mismatches.

SKILL 4 — EVENT EXTRACTION (action within one moment)
Trigger: a verb phrase describing what an entity is doing, distinct from a static relation.
Procedure: one event per distinct action, with participants listed by entity id. `action` should be a canonical English verb-based tag describing the action generically — do not invent narrower categories than the query supports, and do not paraphrase into an action the query didn't state.

SKILL 5 — TEMPORAL SEQUENCING (multi-event ordering)
Trigger: the query describes more than one event AND specifies or implies their order in time (this is the dominant skill for TRAKE-type queries with numbered/sequential sub-events).
Procedure: only create temporal_constraints between events that already exist from Skill 4. Use "before"/"after"/"during"/"overlaps" strictly as the query's wording supports — if order is implied only by narrative sequence (e.g. numbered steps), treat that as "after" chains unless the query says otherwise.

SKILL 6 — SPATIAL ORDERING (rank among peers)
Trigger: the query distinguishes an entity from similar entities by spatial position along an axis (leftmost, second from the right, closest to camera, etc.), NOT by time.
Procedure: this always produces TWO separate outputs — an order_constraint describing the axis/direction, and a selection_constraint giving the rank. Never fold them into one field, since they answer different questions (how to sort vs. which position to pick).

SKILL 7 — NEGATION SWEEP (run last, over everything already extracted)
Trigger: any explicit negation, absence, or exclusion in the query text (words meaning "without", "not", "no one who...", "excluding").
Procedure: find which attribute/relation/event the negation scopes over, and flip its polarity to "NEGATIVE" instead of creating a separate "absence" entity. If the negation refers to something with no clear antecedent entity/relation already extracted, do not fabricate a new one just to negate it — instead note the gap in meta.ambiguous_notes.

SELF-CHECK BEFORE OUTPUT (run silently, do not print your reasoning):
- Does every id used in relations/events/temporal_constraints/order_constraints/selection_constraints exist in entities or events?
- Did I add anything the query did not state or clearly imply? If yes, remove it.
- Did I skip anything the query stated? If yes, add it.
- Is every enum-like field (polarity, axis, direction, temporal relation) filled with one of exactly: polarity ∈ {{"POSITIVE","NEGATIVE"}}; axis ∈ {{"horizontal","vertical","depth"}}; direction ∈ {{"left_to_right","right_to_left","top_to_bottom","bottom_to_top","near_to_far","far_to_near"}}; temporal relation ∈ {{"before","after","during","overlaps"}}?
- Set meta.confidence (0.0–1.0) honestly — lower it whenever a skill above had to guess or a term didn't map cleanly. Use meta.ambiguous_notes (English, one short sentence) for anything you weren't fully sure about; leave it empty string if none.

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
            resp = requests.post(url, json=payload, timeout=10.0, verify=False)
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
