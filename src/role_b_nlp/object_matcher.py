import re
from typing import List, Dict

# Mapping Vietnamese query terms to OpenImages / Faster R-CNN object classes
OBJECT_DICTIONARY: Dict[str, List[str]] = {
    "người": ["person", "man", "woman", "boy", "girl"],
    "ô tô": ["car", "vehicle", "land vehicle", "truck", "bus"],
    "xe máy": ["motorcycle", "bicycle"],
    "chó": ["dog", "carnivore", "mammal"],
    "mèo": ["cat"],
    "bàn": ["table", "desk"],
    "ghế": ["chair", "bench"],
    "cây": ["tree", "plant", "houseplant"],
    "điện thoại": ["mobile phone", "telephone"],
}

def extract_object_keywords(text: str) -> List[str]:
    """
    Extracts physical object names from text query matching detection dictionaries.
    """
    if not text:
        return []
        
    found_objects = set()
    text_lower = text.lower()
    
    for vn_term, en_classes in OBJECT_DICTIONARY.items():
        if vn_term in text_lower:
            found_objects.update(en_classes)
            
    return list(found_objects)

def calculate_object_match_score(target_objects: List[str], detected_classes: List[str]) -> float:
    """
    Computes overlap ratio score between target physical objects in query and detected classes in frame.
    Score ranges from 0.0 to 1.0.
    """
    if not target_objects or not detected_classes:
        return 0.0
        
    target_set = set(obj.lower() for obj in target_objects)
    detected_set = set(obj.lower() for obj in detected_classes)
    
    matched = target_set.intersection(detected_set)
    return len(matched) / len(target_set)
