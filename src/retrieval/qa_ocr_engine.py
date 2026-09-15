"""
QA OCR-First Engine — SigLIP-Guided + Lazy OCR + VLM Verify
=============================================================
Strategy:
  1. Classify question type → OCR-able vs VISUAL-only
  2. [OCR-able] SigLIP candidates are already ranked; pick Top-N frames
     from the RankedCandidate window (best vector → best candidate for OCR)
  3. Run EasyOCR on those frames → extract raw text
  4. Majority vote (confidence-weighted) → top-3 answer candidates
  5. VLM verify ONLY the best single frame (1 API call max) to confirm/pick
  6. [VISUAL-only] Fallback: VLM direct on top-1 frame (unchanged behavior)

Token cost: max 1 VLM call/query (vs ~5-10 before)
OCR accuracy: uses siglip_score to pick frames with clearest content
"""

import os
import io
import re
import base64
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("OFFLINE_CACHE_DIR", "data/processed/cache")) / "qa_ocr"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Question Type Keywords ────────────────────────────────────────────────────

# Câu hỏi yêu cầu đọc ký tự/số trên màn hình → OCR-able
_OCR_KEYWORDS_VI = [
    "số mấy", "số hiệu", "số xe", "biển số", "con số", "ký tự",
    "tên gì", "ghi gì", "ghi là gì", "viết gì", "chữ gì", "tiêu đề",
    "bao nhiêu ký tự", "mã số", "ký hiệu", "tên đường", "tên cửa hàng",
    "bảng hiệu", "biển hiệu", "bảng số", "số điện thoại", "địa chỉ",
]
_OCR_KEYWORDS_EN = [
    "what number", "which number", "what text", "what does it say",
    "what is written", "what sign", "license plate", "registration",
    "what label", "what title", "serial number", "what code",
]

# Câu hỏi thuần visual → VLM trực tiếp
_VISUAL_KEYWORDS = [
    # Colors
    "màu gì", "mau gi", "color", "what color", "màu sắc", "mau sac",
    # Emotion/expression
    "cảm xúc", "cam xuc", "emotion", "biểu cảm", "bieu cam", "feeling",
    # Action/behavior
    "hành động gì", "hanh dong gi", "what is happening", "what are they doing",
    "ai đang làm", "ai dang lam", "đang làm gì", "dang lam gi",
    # Appearance without text
    "trông như thế nào", "trong nhu the nao", "look like",
    # Count (no OCR needed)
    "có bao nhiêu người", "co bao nhieu nguoi", "how many people",
]


def _detect_question_type(question: str) -> str:
    """
    Returns 'OCR' if the question requires reading on-screen text/numbers,
    or 'VISUAL' if it's purely visual (color, emotion, action).
    Default: 'OCR' (err on the side of trying OCR first).
    """
    q_lower = question.lower()
    for kw in _VISUAL_KEYWORDS:
        if kw in q_lower:
            # Check if there's also an OCR indicator overriding
            for ocr_kw in _OCR_KEYWORDS_VI + _OCR_KEYWORDS_EN:
                if ocr_kw in q_lower:
                    return "OCR"
            return "VISUAL"
    for kw in _OCR_KEYWORDS_VI + _OCR_KEYWORDS_EN:
        if kw in q_lower:
            return "OCR"
    # Default: try OCR first (cheaper than VLM)
    return "OCR"


def _get_cache(cache_key: str) -> Optional[str]:
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("answer")
        except Exception:
            pass
    return None


def _set_cache(cache_key: str, answer: str):
    path = CACHE_DIR / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"answer": answer}, f, ensure_ascii=False)


# ── Image loader ──────────────────────────────────────────────────────────────

def _load_image_numpy(video_id: str, frame_idx: int):
    import numpy as np
    from PIL import Image
    try:
        from tools.review_tool import resolve_keyframe_b64
        b64_str, _, _ = resolve_keyframe_b64(
            video_id, frame_idx,
            keyframes_root=os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'raw', 'keyframes'
            ),
            allow_remote=True  # allow remote for QA (need best frames)
        )
        if not b64_str:
            return None
        # b64_str is "data:image/jpeg;base64,..."
        img_data = base64.b64decode(b64_str.split(',')[1])
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        return np.array(img)
    except Exception as e:
        logger.warning(f"[QAOCREngine] Image load failed {video_id}/{frame_idx}: {e}")
        return None


# ── OCR Reader (singleton, reuses LazyOCREngine instance if available) ─────────

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("[QAOCREngine] Initializing EasyOCR (vi, en)...")
            _reader = easyocr.Reader(['vi', 'en'], gpu=False)
        except Exception as e:
            logger.error(f"[QAOCREngine] EasyOCR init failed: {e}")
    return _reader


# ── Core OCR Scan ─────────────────────────────────────────────────────────────

def _ocr_frames(
    candidates_sorted: list,   # List[RankedCandidate] sorted by siglip_score DESC
    top_n: int = 8,
) -> List[Tuple[str, float, str, int]]:
    """
    Run OCR on top-N frames (highest siglip_score = best visual representation).
    Returns list of (text_token, weighted_confidence, video_id, frame_idx).

    SigLIP score determines WHICH frames get OCR — higher score = clearer scene.
    """
    reader = _get_reader()
    if not reader:
        return []

    results = []
    frames_to_scan = candidates_sorted[:top_n]
    logger.info(f"[QAOCREngine] Scanning {len(frames_to_scan)} SigLIP-top frames for OCR...")

    for rc in frames_to_scan:
        vid = rc.candidate.video_id
        fidx = rc.candidate.frame_idx
        siglip_w = rc.score.visual  # Use visual score as OCR frame weight

        img_np = _load_image_numpy(vid, fidx)
        if img_np is None:
            continue

        try:
            detections = reader.readtext(img_np, detail=1)
            for (_, text, conf) in detections:
                text = text.strip()
                if not text or conf < 0.3:
                    continue
                # Weight = OCR confidence × SigLIP visual score (frame quality)
                weighted_conf = conf * (0.5 + 0.5 * siglip_w)
                results.append((text, weighted_conf, vid, fidx))
                logger.debug(f"  [OCR] {vid}/{fidx}: '{text}' (conf={conf:.2f}, w={weighted_conf:.3f})")
        except Exception as e:
            logger.error(f"[QAOCREngine] EasyOCR error on {vid}/{fidx}: {e}")

    return results


# ── Majority Vote ─────────────────────────────────────────────────────────────

def _vote_answer(
    question: str,
    ocr_results: List[Tuple[str, float, str, int]],
) -> List[Tuple[str, float, str, int]]:
    """
    Weighted majority vote on OCR tokens.
    Returns top-3 (candidate_text, total_weight, best_video_id, best_frame_idx).

    Strategy: accumulate confidence-weighted votes per unique text token.
    Tokens that appear in multiple frames get higher weight.
    """
    # Token-level vote (exact text as key)
    vote_map: Dict[str, Tuple[float, str, int]] = {}

    for text, weight, vid, fidx in ocr_results:
        normalized = text.strip()
        if not normalized:
            continue
        if normalized in vote_map:
            prev_w, prev_vid, prev_fidx = vote_map[normalized]
            # Keep frame with highest weight as the "best frame"
            new_w = prev_w + weight
            best_frame = (vid, fidx) if weight > prev_w else (prev_vid, prev_fidx)
            vote_map[normalized] = (new_w, best_frame[0], best_frame[1])
        else:
            vote_map[normalized] = (weight, vid, fidx)

    # Sort by total weight descending
    ranked = sorted(vote_map.items(), key=lambda x: x[1][0], reverse=True)

    top3 = []
    for text, (w, vid, fidx) in ranked[:3]:
        top3.append((text, w, vid, fidx))

    logger.info(f"[QAOCREngine] Vote top-3: {[(t, round(w, 3)) for t, w, _, _ in top3]}")
    return top3


# ── VLM Verify ───────────────────────────────────────────────────────────────

def _vlm_verify(
    question: str,
    answer_candidates: List[str],
    video_id: str,
    frame_idx: int,
) -> str:
    """
    Ask VLM to confirm which candidate is correct, given the frame image.
    Only called ONCE per query on the single best frame.

    If VLM unavailable → return top-1 OCR candidate directly.
    """
    try:
        import google.generativeai as genai
        from PIL import Image

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_GEMINI")
        # Fallback: read from .env.example
        if not api_key:
            for env_file in [".env", ".env.example"]:
                p = Path(env_file)
                if p.exists():
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip() in ("API_GEMINI", "GEMINI_API_KEY"):
                                api_key = v.strip().strip('"').strip("'")
                                break
                if api_key:
                    break

        if not api_key:
            logger.warning("[QAOCREngine] No API key → using top-1 OCR candidate.")
            return answer_candidates[0] if answer_candidates else "Không tìm thấy đáp án."

        # Load image
        from tools.review_tool import resolve_keyframe_b64
        b64_str, _, _ = resolve_keyframe_b64(
            video_id, frame_idx,
            keyframes_root=os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'raw', 'keyframes'
            ),
            allow_remote=True
        )
        if not b64_str:
            return answer_candidates[0] if answer_candidates else "Không tìm thấy đáp án."

        img_data = base64.b64decode(b64_str.split(',')[1])
        image = Image.open(io.BytesIO(img_data)).convert('RGB')

        import requests
        requests.packages.urllib3.disable_warnings()
        if not hasattr(requests.Session, '_ssl_patched'):
            old_req = requests.Session.request
            def _no_ssl(self, method, url, **kwargs):
                kwargs['verify'] = False
                return old_req(self, method, url, **kwargs)
            requests.Session.request = _no_ssl
            requests.Session._ssl_patched = True

        genai.configure(api_key=api_key, transport='rest')
        model = genai.GenerativeModel("gemini-3.5-flash-lite")

        candidates_str = "\n".join(
            f"  {i+1}. \"{c}\"" for i, c in enumerate(answer_candidates)
        )
        prompt = f"""You are a Visual QA verifier. Look at the image carefully.
Question: "{question}"

OCR detected these candidate answers from the image:
{candidates_str}

Which candidate is the CORRECT answer to the question?
If a candidate matches text visible in the image, select it exactly.
If none match, describe what you see that answers the question.
Respond with ONLY the answer text in Vietnamese. No explanation."""

        response = model.generate_content([prompt, image])
        answer = response.text.strip()
        logger.info(f"[QAOCREngine] VLM verified answer: '{answer}'")
        return answer

    except Exception as e:
        logger.error(f"[QAOCREngine] VLM verify failed: {e}")
        return answer_candidates[0] if answer_candidates else "Lỗi xác nhận VLM."


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question_ocr_first(
    question: str,
    ranked_candidates: list,  # List[RankedCandidate] — already sorted by siglip_score
    top_n_ocr: int = 8,
) -> str:
    """
    Main entry point for QA answering.

    Args:
        question: The user question (Vietnamese or English)
        ranked_candidates: List of RankedCandidate from HybridSearcher,
                           sorted by score.final DESC (= visual quality DESC).
        top_n_ocr: How many top SigLIP frames to scan with OCR.

    Returns:
        A string answer.
    """
    if not ranked_candidates:
        return "Không tìm thấy kết quả để trả lời."

    top_cand = ranked_candidates[0]

    # ── Cache check ──────────────────────────────────────────────────────────
    cache_key = hashlib.sha1(
        f"QAOCR_{question}_{top_cand.candidate.video_id}_{top_cand.candidate.frame_idx}".encode()
    ).hexdigest()
    cached = _get_cache(cache_key)
    if cached:
        logger.info(f"[QAOCREngine] Cache HIT → '{cached}'")
        return cached

    # ── Classify question ────────────────────────────────────────────────────
    q_type = _detect_question_type(question)
    logger.info(f"[QAOCREngine] Question type: {q_type} | '{question[:80]}'")

    # ── Path A: VISUAL-only → VLM direct ────────────────────────────────────
    if q_type == "VISUAL":
        answer = _vlm_verify(question, [], top_cand.candidate.video_id, top_cand.candidate.frame_idx)
        _set_cache(cache_key, answer)
        return answer

    # ── Path B: OCR-first ────────────────────────────────────────────────────
    # Step 1: OCR on top-N frames (sorted by siglip_score = best visual quality)
    ocr_results = _ocr_frames(ranked_candidates, top_n=top_n_ocr)

    if not ocr_results:
        logger.warning("[QAOCREngine] OCR returned no results → fallback to VLM direct")
        answer = _vlm_verify(question, [], top_cand.candidate.video_id, top_cand.candidate.frame_idx)
        _set_cache(cache_key, answer)
        return answer

    # Step 2: Majority vote → top-3 candidates
    top3 = _vote_answer(question, ocr_results)

    if not top3:
        answer = _vlm_verify(question, [], top_cand.candidate.video_id, top_cand.candidate.frame_idx)
        _set_cache(cache_key, answer)
        return answer

    # Step 3: VLM verify top-1 frame (best weight frame from voting winner)
    candidate_texts = [t for t, _, _, _ in top3]
    best_vid = top3[0][2]
    best_fidx = top3[0][3]

    answer = _vlm_verify(question, candidate_texts, best_vid, best_fidx)
    _set_cache(cache_key, answer)
    return answer
