import re
from typing import Optional

def classify_question(question_text: str) -> Optional[str]:
    """
    Phân loại câu hỏi Q&A trả về chuỗi str thực dụng ("COUNT", "COLOR", "TEXT_OCR", "ACTION", "GENERAL").
    """
    if not question_text:
        return "GENERAL"
        
    text = question_text.lower()
    
    if re.search(r'\b(bao nhiêu|có mấy|bao nhieu|co may|how many|count|đếm)\b', text):
        return "COUNT"

    if re.search(r'\b(màu gì|mau gi|màu sắc|what color|color)\b', text):
        return "COLOR"

    if re.search(r'\b(biển số|chữ gì|ghi chữ|text|sign|bảng hiệu|đọc được)\b', text):
        return "TEXT_OCR"

    if re.search(r'\b(đang làm gì|hành động gì|nhảy|chạy|đi bộ|doing|action)\b', text):
        return "ACTION"

    return "GENERAL"
