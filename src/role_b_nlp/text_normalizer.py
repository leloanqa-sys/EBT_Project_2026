import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Role B Contract Implementation:
    Standardizes input raw text query:
    1. Normalizes unicode (NFC format).
    2. Converts to lowercase.
    3. Strips punctuation and excessive whitespace.
    """
    if not text:
        return ""
    
    # Unicode normalization
    normalized = unicodedata.normalize('NFC', text)
    
    # Lowercase
    normalized = normalized.lower()
    
    # Replace newlines and tabs with spaces
    normalized = re.sub(r'[\r\n\t]+', ' ', normalized)
    
    # Remove unwanted punctuation while keeping alphanumeric and Vietnamese chars
    normalized = re.sub(r'[^\w\s]', ' ', normalized, flags=re.UNICODE)
    
    # Remove multi-spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized
