import os
import sys
import re
import json
import base64
import glob
import requests

# Ensure project root is in sys.path regardless of execution method
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, Dict, Any, List
from src.common.schemas import CandidateFrame, SubmissionOutput, SubmissionItem


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

def get_detected_objects_for_frame(video_id: str, frame_idx: int, objects_root: str = "data/raw/objects") -> List[str]:
    """
    Fetches object detection entities from Faster R-CNN JSON for a specific frame.
    """
    vid_dir = os.path.join(objects_root, video_id)
    if not os.path.exists(vid_dir):
        return []

    # Try mapping frame_idx or sequential file names like 001.json
    json_path = os.path.join(vid_dir, f"{frame_idx:03d}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(vid_dir, f"{frame_idx}.json")

    if not os.path.exists(json_path):
        files = glob.glob(os.path.join(vid_dir, "*.json"))
        if files:
            json_path = files[0]

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("detection_class_entities", [])[:15]
        except Exception:
            pass

    return []

def audit_frame_with_gemini_vlm(
    query_text: str,
    video_id: str,
    frame_idx: int,
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "models/gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    READ-ONLY Audit Function:
    Calls Gemini API to evaluate match probability for a candidate frame.
    STRICT CONSTRAINT: DOES NOT MUTATE OR ALTER PIPELINE OUTPUTS / SCORES / RANKS!
    """
    if not api_key:
        api_key = load_gemini_api_key()

    if not api_key:
        return {
            "status": "SKIPPED",
            "match_probability": 0.0,
            "explanation": "No GEMINI_API_KEY / API_GEMINI found in .env or environment.",
            "is_match": False
        }

    # Ensure model string is formatted correctly for endpoint
    model_name = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"

    # Path A: Image-based Vision Audit (if JPG/PNG exists)
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            f"You are a strict read-only visual verification auditor for a video retrieval competition. "
            f"Target Query: '{query_text}'. "
            f"Evaluate if this image matches the target description. "
            f"Respond ONLY in valid JSON with keys: "
            f"\"is_match\": (boolean), \"match_probability\": (float between 0.0 and 1.0), \"explanation\": (short 1-sentence reason)."
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }],
            "generationConfig": {"response_mime_type": "application/json"}
        }
    # Path B: Metadata & Detected Objects Audit (if image not downloaded locally)
    else:
        detected_objs = get_detected_objects_for_frame(video_id, frame_idx)
        objs_str = ", ".join(detected_objs) if detected_objs else "None detected"

        prompt = (
            f"You are a strict read-only verification auditor for a video retrieval competition. "
            f"Target Query: '{query_text}'. "
            f"Candidate Metadata: video_id='{video_id}', frame_idx={frame_idx}, detected_objects=[{objs_str}]. "
            f"Evaluate if these detected visual objects match the target description. "
            f"Respond ONLY in valid JSON with keys: "
            f"\"is_match\": (boolean), \"match_probability\": (float between 0.0 and 1.0), \"explanation\": (short 1-sentence reason)."
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }

    try:
        resp = requests.post(url, json=payload, timeout=12)
        if resp.status_code == 429:
            return {
                "status": "QUOTA_EXCEEDED",
                "match_probability": 0.0,
                "explanation": "Gemini API Free Tier rate limit reached (HTTP 429). Quota will reset shortly.",
                "is_match": False
            }

        if resp.status_code != 200:
            return {
                "status": "API_ERROR",
                "match_probability": 0.0,
                "explanation": f"Gemini API status {resp.status_code}: {resp.text[:100]}",
                "is_match": False
            }

        res_json = resp.json()
        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)

        return {
            "status": "SUCCESS",
            "is_match": bool(parsed.get("is_match", False)),
            "match_probability": float(parsed.get("match_probability", 0.0)),
            "explanation": str(parsed.get("explanation", "")),
            "video_id": video_id,
            "frame_idx": frame_idx
        }
    except Exception as e:
        return {
            "status": "EXCEPTION",
            "match_probability": 0.0,
            "explanation": str(e),
            "is_match": False
        }

def run_read_only_vlm_audit(
    query_id: str,
    query_text: str,
    ranked_items: List[SubmissionItem],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Executes Read-Only Gemini VLM Audit on Top N submission items.
    Guarantees zero mutation on submission items or scores.
    Returns audit summary and verdict report.
    """
    api_key = load_gemini_api_key()
    audit_reports = []
    probabilities = []

    for item in ranked_items[:top_n]:
        res = audit_frame_with_gemini_vlm(
            query_text=query_text,
            video_id=item.video_id,
            frame_idx=item.frame_id,
            api_key=api_key
        )
        res["rank"] = item.rank
        res["retrieval_confidence"] = round(item.confidence_score, 4)
        audit_reports.append(res)
        if res.get("status") == "SUCCESS":
            probabilities.append(res["match_probability"])

    avg_prob = sum(probabilities) / len(probabilities) if probabilities else 0.0
    top1_match = audit_reports[0].get("is_match", False) if audit_reports else False
    top1_prob = audit_reports[0].get("match_probability", 0.0) if audit_reports else 0.0

    verdict = "PASS" if (top1_match or top1_prob >= 0.5 or avg_prob >= 0.4) else "FLAGGED_LOW_MATCH"

    return {
        "query_id": query_id,
        "query_text": query_text,
        "verdict": verdict,
        "top1_match_probability": round(top1_prob, 4),
        "avg_top_n_probability": round(avg_prob, 4),
        "item_audits": audit_reports
    }

if __name__ == "__main__":
    print("--- [READ-ONLY Gemini VLM Audit Module] ---")
    key = load_gemini_api_key()
    print(f"Gemini API Key Loaded: {'YES (' + key[:8] + '...)' if key else 'NO'}")
    
    # Test audit on sample item
    mock_item = SubmissionItem(rank=1, video_id="L21_V001", frame_id=90, confidence_score=0.25)
    audit_res = run_read_only_vlm_audit("sample_audit", "Tìm cảnh tòa nhà cao tầng", [mock_item])
    print(json.dumps(audit_res, indent=2, ensure_ascii=False))
