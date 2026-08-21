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

def test_single_query_audit():
    api_key = load_gemini_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
    
    system_instruction = (
        "Ban la Giam khao AI doc lap cho cuoc thi Video Retrieval AIC 2026.\n"
        "Nhiem vu: Tham dinh tinh dung/sai cua video ung vien doi chieu voi yeu cau de bai.\n"
        "Phan hoi DUY NHAT JSON: {\"verdict\": \"MATCH\"|\"UNCERTAIN\"|\"MISMATCH\", \"confidence\": float, \"factual_answer\": string, \"reason\": string}"
    )
    
    sample_prompt = (
        "DE THI KIS (query-p1-6-kis): Doan clip bat dau bang canh 1 nguoi dau bep dat mon goi cuon chay bay tren dia, nhan gom rau xanh cuon tron va dau hu, goi trong banh trang mau vang va tim. Dia duoc trang tri them la xanh va hoa pansy tim-vang.\n"
        "Ung vien: Video L26_V056, Frame 6656. Thuc the phat hien: [dish, plate, food, spring roll, tofu, yellow, purple, flower, chef].\n"
        "Hay tham dinh tinh dung sai."
    )
    
    payload = {
        "contents": [{"parts": [{"text": f"{system_instruction}\n\n{sample_prompt}"}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 150,
            "responseMimeType": "application/json"
        }
    }
    
    t0 = time.time()
    res = requests.post(url, json=payload, timeout=20, verify=False)
    elapsed = time.time() - t0
    
    print("Status:", res.status_code, f"({elapsed:.2f}s)")
    if res.status_code == 200:
        data = res.json()
        text = data['candidates'][0]['content']['parts'][0]['text']
        usage = data.get('usageMetadata', {})
        print("Model Response:\n", text)
        print("Token Usage:", usage)
        in_tok = usage.get('promptTokenCount', 0)
        out_tok = usage.get('candidatesTokenCount', 0)
        cost_usd = (in_tok * 0.075 + out_tok * 0.3) / 1000000
        cost_vnd = cost_usd * 25500
        print(f"\nChi phi 1 luot tham dinh: ${cost_usd:.6f} USD (~{cost_vnd:.2f} VND)")
        print(f"Tong uoc tinh cho toan bo 24 cau (~72 luot): {cost_vnd * 72:.2f} VND (Chi chiem ~{(cost_vnd * 72 / 2000)*100:.1f}% cua 2.000 VND)")

if __name__ == '__main__':
    test_single_query_audit()
