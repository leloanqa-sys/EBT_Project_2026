import os
import json
import hashlib
import requests
import re
from typing import List

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

    for env_file in [".env", ".env.example"]:
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

# Common OpenImages V4 / Faster R-CNN target object keyword mappings (English & Vietnamese)
COMMON_OBJECT_KEYWORD_MAP = {
    # Ball / Sports
    "soccer ball": ["Ball", "Football"],
    "football": ["Ball", "Football"],
    "ball": ["Ball"],
    "bóng": ["Ball"],
    "quả bóng": ["Ball"],
    "bóng đá": ["Ball", "Football"],
    # People
    "person": ["Person"],
    "man": ["Man", "Person"],
    "woman": ["Woman", "Person"],
    "boy": ["Boy", "Person"],
    "girl": ["Girl", "Person"],
    "người": ["Person"],
    "đàn ông": ["Man", "Person"],
    "phụ nữ": ["Woman", "Person"],
    "trẻ em": ["Person", "Boy", "Girl"],
    # Vehicles
    "car": ["Car", "Vehicle", "Land vehicle"],
    "vehicle": ["Vehicle", "Land vehicle"],
    "motorcycle": ["Motorcycle", "Vehicle"],
    "bicycle": ["Bicycle", "Vehicle"],
    "bus": ["Bus", "Vehicle"],
    "truck": ["Truck", "Vehicle"],
    "boat": ["Boat", "Vehicle"],
    "ô tô": ["Car", "Vehicle"],
    "xe máy": ["Motorcycle", "Vehicle"],
    "xe đạp": ["Bicycle", "Vehicle"],
    # Clothes / Accessories
    "clothing": ["Clothing"],
    "shirt": ["Clothing"],
    "dress": ["Clothing"],
    "hat": ["Hat", "Clothing"],
    "glasses": ["Glasses"],
    "áo": ["Clothing"],
    "quần": ["Clothing"],
    "mũ": ["Hat"],
    "kính": ["Glasses"],
    # Nature / Outdoor
    "tree": ["Tree"],
    "flower": ["Flower"],
    "plant": ["Plant"],
    "building": ["Building"],
    "house": ["House", "Building"],
    "cây": ["Tree"],
    "hoa": ["Flower"],
    "tòa nhà": ["Building"],
    "nhà": ["House", "Building"],
    # Animals
    "dog": ["Dog", "Animal"],
    "cat": ["Cat", "Animal"],
    "chó": ["Dog", "Animal"],
    "mèo": ["Cat", "Animal"],
}

def clean_instruction_words(text: str) -> str:
    """Strips command/instruction noise words like 'FIND A', 'SEARCH FOR', 'TÌM'."""
    pattern = r'^\s*(?:find\s+(?:a|an|the)?|search\s+for|look\s+for|show\s+me|where\s+is|tìm\s+(?:kiếm)?|hãy\s+tìm)\s+'
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else text

def extract_target_objects_fallback(query_text: str) -> List[str]:
    """Local rule-based fallback for target object extraction when API is unavailable or rate-limited."""
    text_lower = clean_instruction_words(query_text).lower()
    extracted = set()

    for kw, labels in COMMON_OBJECT_KEYWORD_MAP.items():
        if kw in text_lower:
            for l in labels:
                extracted.add(l)

    return list(extracted)

GEMINI_MODEL_CASCADE = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]

def extract_target_objects(query_text: str) -> List[str]:
    """
    Extracts target objects mentioned in Vietnamese/English query.
    Cascade through GEMINI_MODEL_CASCADE; falls back to local rule-based extraction if all models fail/hit quota.
    """
    if not query_text.strip():
        return []

    clean_text = clean_instruction_words(query_text)

    cached = _read_cache(clean_text, "objects")
    if cached is not None:
        return cached.get("target_objects_english", [])

    api_key = load_gemini_api_key()
    if not api_key:
        return extract_target_objects_fallback(clean_text)

    prompt = (
        f"You are a computer vision AI assistant. "
        f"Extract all physical target objects mentioned in the query (which may be in Vietnamese or English): '{clean_text}'. "
        f"Map or translate them to standard English object detection labels (such as Faster R-CNN, OpenImages V4, COCO). "
        f"Examples: 'người' or 'a man speaking' -> 'Person', 'soccer ball' or 'bóng đá' -> 'Ball', 'car' -> 'Car', 'building' -> 'Building'. "
        f"Return ONLY a valid JSON object with key 'target_objects_english' (array of strings)."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    # Model Fallback Cascade
    for model_name in GEMINI_MODEL_CASCADE:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=2.5)
            if resp.status_code == 200:
                res_json = resp.json()
                raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(raw_text)
                objs = parsed.get("target_objects_english", [])
                if isinstance(objs, list) and objs:
                    _write_cache(clean_text, "objects", {"target_objects_english": objs})
                    return objs
        except Exception:
            continue

    # Final Local Rule-based Fallback
    fallback_objs = extract_target_objects_fallback(clean_text)
    if fallback_objs:
        _write_cache(clean_text, "objects", {"target_objects_english": fallback_objs})
    return fallback_objs

def decompose_events(query_text: str) -> List[str]:
    """
    Decomposes a query into multiple sequential sub-events for TRAKE.
    Cascade through GEMINI_MODEL_CASCADE; falls back to regex parser if all fail.
    """
    if not query_text.strip():
        raise ValueError("Empty query")

    clean_text = clean_instruction_words(query_text)

    cached = _read_cache(clean_text, "events")
    if cached is not None:
        return cached.get("sub_events", [])

    api_key = load_gemini_api_key()
    if not api_key:
        raise ValueError("No API key")

    prompt = (
        f"Decompose the following video retrieval query (which may be in Vietnamese or English) into a sequential sequence of sub-events in chronological order. "
        f"Query: '{clean_text}'. "
        f"Return ONLY a valid JSON object with key 'sub_events' (array of strings)."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    for model_name in GEMINI_MODEL_CASCADE:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=2.5)
            if resp.status_code == 200:
                res_json = resp.json()
                raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(raw_text)
                events = parsed.get("sub_events", [])
                if isinstance(events, list) and events:
                    _write_cache(clean_text, "events", {"sub_events": events})
                    return events
        except Exception:
            continue

    raise ValueError("All Gemini models failed or exceeded quota")
