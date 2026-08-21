import os
import sys
import json
import time
import requests
import urllib3
from pathlib import Path
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.role_b_nlp.gemini_nlp_engine import load_gemini_api_key

def test_flash_lite():
    api_key = load_gemini_api_key()
    model = "gemini-3.1-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": "Hello, reply with JSON: {\"status\": \"ok\", \"model\": \"gemini-3.1-flash-lite\"}"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60, "responseMimeType": "application/json"}
    }
    
    t0 = time.time()
    res = requests.post(url, json=payload, timeout=10, verify=False)
    elapsed = time.time() - t0
    print(f"Status: {res.status_code} ({elapsed:.2f}s)")
    if res.status_code == 200:
        data = res.json()
        print("Response:", data['candidates'][0]['content']['parts'][0]['text'])
        usage = data.get('usageMetadata', {})
        print("Token usage:", usage)
        cost_usd = (usage.get('promptTokenCount', 0) * 0.025 + usage.get('candidatesTokenCount', 0) * 0.10) / 1000000
        print(f"Cost: ${cost_usd:.8f} USD (~{cost_usd * 25500:.4f} VND)")

if __name__ == '__main__':
    test_flash_lite()
