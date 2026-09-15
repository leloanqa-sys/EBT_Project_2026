"""
Gemini NLP Query Parser — Phase 4
===================================
Transforms raw Vietnamese queries into structured QueryIR objects.

Design constraints:
  - Gemini ONLY acts as a compiler/planner.
  - Gemini MUST output JSON matching QueryIR schema (Structured Output).
  - Gemini does NOT see images, does NOT rank frames.
  - Each Gemini call is cached to disk → zero re-calls for same query.
  - API key must be set via env: GOOGLE_API_KEY (or loaded from .env.example)

Phase 4 additions (v2):
  - query_class field: Q_VISUAL | Q_ENTITY | Q_METADATA | Q_COMPOSITE
  - REMOVED reasoning_chain (bloat, slows down parsing).
  - STRICT constraint on visual_targets (no named entities to prevent averaging dilution).
  - parser_mode explicitly logged as GEMINI or FALLBACK.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional

from src.common.schemas import (
    QueryIR, VisualTarget, TextTarget
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/cache/nlp_ir")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Monkey patch requests to ignore SSL (since Windows local env blocks it)
import requests
requests.packages.urllib3.disable_warnings()
_old_request = requests.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _old_request(self, method, url, **kwargs)
requests.Session.request = _new_request

GEMINI_MODEL = "gemini-3.5-flash-lite"  # Lite model equivalent: sufficient for structured JSON NLP parsing

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — FROZEN (phase4_v2)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a multi-modal query decomposition engine for a Vietnamese video retrieval system.
Your job: transform a Vietnamese query into a structured JSON object.

═══ ARCHITECTURE & CONTEXT ═══
Our system uses 3 independent retrieval engines:
1. SigLIP (Image-Text Dual Encoder): Understands rich, descriptive natural language. It handles adjectives (colors, sizes), actions, and context best when given a fluent English sentence.
2. Object Detector (Faster R-CNN): Understands pure, physical bounding box nouns (COCO classes). It does NOT understand actions or colors.
3. OCR/ASR Engine: Finds exact text strings displayed on screen or spoken.

═══ RULES ═══
1. query_class: Classify the query as exactly ONE of:
   Q_VISUAL    — primarily visual (scene, action, appearance).
   Q_ENTITY    — named entity (person name, place name, organization, brand, product).
   Q_METADATA  — metadata-heavy (title, credits, on-screen text, captions).
   Q_COMPOSITE — Contains both independently useful visual and textual signals.

2. dense_caption_en: The input for SigLIP.
   - Translate the visual scene into a rich, fluent English caption.
   - You MUST include ALL visual details: adjectives (pink, tall), verbs (teaching, running), and nouns.
   - Example: "A woman wearing glasses and a pink ao dai is teaching about verb tenses."
   - Do NOT output keywords. Output a natural sentence.

3. visual_targets: (MULTI-TARGET INTERSECT)
   - Break down the dense_caption_en into 1-3 independent visual elements.
   - This is CRUCIAL to force the visual engine not to miss fine details (like flowers, small items).
   - Example: For "A chef placing vegetarian spring rolls on a plate decorated with purple-yellow pansy flowers", output:
     [{{"concept": "A chef plating vegetarian spring rolls", "weight": 0.8}}, {{"concept": "Purple and yellow pansy flowers on a plate", "weight": 1.0}}]
   - Set weight=1.0 for the most crucial fine-grained details, 0.5-0.8 for background/generic objects.

4. object_targets: The input for the Object Detector.
   - You MUST ONLY extract physical nouns that EXACTLY match one of the valid classes listed below.
   - Do NOT extract synonyms. If you want "automobile", you must output "car" (if "car" is in the list).
   - CRITICAL: Do NOT extract colors, verbs, or abstract concepts here.
   - VALID CLASSES (Choose ONLY from these, or leave empty if none match):
   {valid_classes}

5. text_targets: The input for Text Search (Video Title, Description, ASR, OCR).
   - Extract highly specific names, dishes, locations, or unique nouns from the query (e.g., "gỏi cuốn chay", "đậu hũ", "London Zoo").
   - DO NOT extract generic visual words (e.g., "người", "màu đỏ", "bên trái"). Only extract words that likely appear in the video's Title, Audio, or on-screen text.
   - You MUST keep these terms in their original language (e.g., Vietnamese) to match the database titles exactly.
   - Set weight = 0.9 for highly unique names/titles, and 0.4 for common topic words.

6. object_constraints: (EXPERIMENTAL TIER-2 FILTER)
   - If the query specifies an EXACT NUMBER of objects (e.g., "5 people", "3 red hats", "2 dogs"), extract it here.
   - WARNING: You MUST translate Vietnamese words for numbers ("một", "hai", "ba", "bốn"...) into digits! For example, "hai người phụ nữ" -> [{{"class_name": "person", "min_count": 2}}]
   - If the query implies spatial conditions (e.g. "sitting between", "on the left"), add it to spatial_condition.
   - class_name MUST be from the VALID CLASSES list.
   - Example: [{{"class_name": "person", "min_count": 5, "spatial_condition": "none"}}]
   - If no explicit count is mentioned, leave empty [].

7. temporal_sequence: Events in chronological order (TRAKE queries only). 
   - You MUST translate each event into fluent English. SigLIP only understands English.
   - If query_type != TRAKE, return [].

Output ONLY valid JSON with the exact keys shown below.
"""

_USER_PROMPT_TEMPLATE = """\
Query Type: {query_type}
Raw Text (Vietnamese): {raw_text}

Decompose this into JSON with these exact keys:
{{
  "query_class": "Q_VISUAL | Q_ENTITY | Q_METADATA | Q_COMPOSITE",
  "dense_caption_en": "...",
  "visual_targets": [{{"concept": "...", "weight": 0.0-1.0}}],
  "text_targets": [{{"term": "...", "source_priority": ["ASR","OCR","TITLE","DESCRIPTION"], "weight": 0.0-1.0}}],
  "object_targets": ["..."],
  "temporal_sequence": ["..."],
  "parser_confidence": 0.0-1.0,
  "ambiguity_note": "..."
}}
"""


def _cache_key(query_id: str, raw_text: str, prompt_version: str) -> str:
    """Include prompt_version in cache key so prompt changes invalidate old cache."""
    h = hashlib.sha1(f"{prompt_version}:{query_id}:{raw_text}".encode()).hexdigest()[:12]
    return h


def _load_from_cache(cache_key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_to_cache(cache_key: str, data: dict):
    path = CACHE_DIR / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_api_key() -> Optional[str]:
    # Check multiple possible env vars
    for key_name in ["API_GEMINI", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key_name)
        if val and val.strip():
            return val.strip()
    
    # Fallback to .env and .env.example
    for env_filename in [".env", ".env.example"]:
        env_path = Path(env_filename)
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    if k in ["API_GEMINI", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"]:
                        return v.strip().strip('"').strip("'")
    return None


def parse_query_to_ir(
    query_id: str,
    raw_text: str,
    query_type: str = "KIS",
    force_fallback: bool = False,
) -> QueryIR:
    """
    Parse a Vietnamese query into a structured QueryIR.
    Result is disk-cached: each unique (query, prompt_version) incurs exactly 1 API call.

    Logs:
      - [GeminiNLP] Cache HIT / MISS
      - [GeminiNLP] API call (model, tokens if available)
      - [GeminiNLP] Fallback reason
    """
    from src.common.config import GEMINI_PROMPT_VERSION

    if force_fallback:
        logger.info(f"[GeminiNLP] force_fallback=True for query_id={query_id}")
        return _fallback_ir(query_id, raw_text, query_type)

    ck = _cache_key(query_id, raw_text, GEMINI_PROMPT_VERSION)
    cached = _load_from_cache(ck)

    if cached:
        logger.info(f"[GeminiNLP] Cache HIT for query_id={query_id} (key={ck})")
        cached["query_id"] = query_id
        cached["raw_text"] = raw_text
        cached["query_type"] = query_type
        # Explicitly set parser_mode if missing from old cache
        if "parser_mode" not in cached:
            cached["parser_mode"] = "GEMINI"
        return QueryIR(**cached)

    # ── Check API key ──────────────────────────────────────────────────────
    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            f"[GeminiNLP] GOOGLE_API_KEY not set → fallback IR (SigLIP-only mode) "
            f"for query_id={query_id}"
        )
        return _fallback_ir(query_id, raw_text, query_type)

    # ── Call Gemini ───────────────────────────────────────────────────────
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key, transport='rest')
        # Load valid classes for the prompt
        try:
            with open("data/taxonomy/detected_classes_cache.json", "r", encoding="utf-8") as f:
                taxonomy = json.load(f)
                valid_classes_str = ", ".join(taxonomy.get("classes", []))
        except Exception as e:
            logger.warning(f"[GeminiNLP] Could not load taxonomy: {e}")
            valid_classes_str = "person, car, animal, clothing, face, etc."

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=_SYSTEM_PROMPT.format(valid_classes=valid_classes_str),
            generation_config=genai.GenerationConfig(
                temperature=0.1,                    # Near-deterministic structural output
                response_mime_type="application/json"
            )
        )
        prompt = _USER_PROMPT_TEMPLATE.format(query_type=query_type, raw_text=raw_text)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                parsed = json.loads(response.text)
                break
            except Exception as loop_e:
                import time
                if "429" in str(loop_e) or "Too Many Requests" in str(loop_e):
                    delay = 2.0 * (2 ** attempt)
                    logger.warning(f"[GeminiNLP] 429 Rate Limit. Retrying in {delay}s...")
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise

        # Stamp metadata
        parsed["query_id"] = query_id
        parsed["raw_text"] = raw_text
        parsed["query_type"] = query_type
        parsed["parser_mode"] = "GEMINI"

        # Remove reasoning_chain if model hallucinates it despite schema
        if "reasoning_chain" in parsed:
            del parsed["reasoning_chain"]

        # Log usage if available
        usage = getattr(response, "usage_metadata", None)
        if usage:
            logger.info(
                f"[GeminiNLP] API call for query_id={query_id} | model={GEMINI_MODEL} | "
                f"prompt_tokens={getattr(usage, 'prompt_token_count', '?')} | "
                f"output_tokens={getattr(usage, 'candidates_token_count', '?')} | "
                f"cached_to={ck}.json"
            )
        else:
            logger.info(
                f"[GeminiNLP] API call for query_id={query_id} | model={GEMINI_MODEL} | "
                f"cached_to={ck}.json"
            )

        _save_to_cache(ck, parsed)
        return QueryIR(**parsed)

    except Exception as e:
        import traceback
        logger.error(f"[GeminiNLP] API error for query_id={query_id}: {repr(e)}\n{traceback.format_exc()} → fallback IR")
        return _fallback_ir(query_id, raw_text, query_type)


def _fallback_ir(query_id: str, raw_text: str, query_type: str) -> QueryIR:
    """
    Fallback when Gemini is unavailable or API key missing.
    Creates a minimal IR using the raw text as a single visual target.
    This is Experiment A (Raw Query → SigLIP) behavior.
    """
    return QueryIR(
        query_id=query_id,
        raw_text=raw_text,
        query_type=query_type,
        query_class="Q_COMPOSITE",
        dense_caption_en=raw_text[:200],
        text_targets=[],
        object_targets=[],
        temporal_sequence=[],
        parser_mode="FALLBACK",
        parser_confidence=0.3,
        ambiguity_note="Fallback mode: Gemini NLP unavailable. SigLIP-only (Exp A behavior)."
    )
