# src/role_c_logic/synonyms_map.py

from typing import Dict, List, Set, Tuple
import logging
from src.role_c_logic.taxonomy_loader import TAXONOMY_CLASSES_SET

logger = logging.getLogger(__name__)

# Bảng tra cứu (Lookup table) tĩnh 1-1. (TẦNG 2)
# Chứa các alias không trùng tên thẳng với TAXONOMY_CLASSES_SET.
CLASS_SYNONYMS_MAP: Dict[str, List[str]] = {
    # 1. Nhóm Người (Trong DB chỉ có "person", không có "man", "woman")
    # Đã loại bỏ "human face" và "clothing" để tránh gánh điểm ảo khi check áo màu sắc hoặc bối cảnh
    "man":    ["person", "man", "human head"],
    "woman":  ["person", "woman", "human head"],
    "boy":    ["person", "boy", "human head"],
    "girl":   ["person", "girl", "human head"],
    
    # 2. Nhóm Trang phục
    "shirt":    ["shirt", "clothing"],
    "pants":    ["trousers", "clothing"],
    "footwear": ["footwear", "clothing"],
    
    # 3. Nhóm Đồ vật & Nội thất
    "computer": ["laptop", "computer monitor", "computer keyboard", "computer mouse"], 
}


# =====================================================================
# KHỐI RÀO LỖI KIẾN TRÚC (IMPORT-TIME INVARIANT TESTS)
# =====================================================================

def _verify_invariants():
    """
    Rào lỗi tự kiểm chứng. Đảm bảo các nhóm thực thể loại trừ nhau 
    không bị nhiễm chéo (cross-contamination).
    """
    exclusion_rules: List[Tuple[str, str, Set[str]]] = [
        ("man", "woman", {"person", "human head"}),
        ("boy", "girl",  {"person", "human head"}),
        ("man", "boy",   {"person", "human head"}),
        ("woman", "girl",{"person", "human head"}),
        
        ("shirt", "pants",    {"clothing"}),
        ("shirt", "footwear", {"clothing"}),
        ("pants", "footwear", {"clothing"}),
        
        ("car", "bicycle", {"vehicle", "land vehicle"}),
    ]
    
    for target_a, target_b, permitted in exclusion_rules:
        if target_a in CLASS_SYNONYMS_MAP and target_b in CLASS_SYNONYMS_MAP:
            set_a = set(CLASS_SYNONYMS_MAP[target_a])
            set_b = set(CLASS_SYNONYMS_MAP[target_b])
            illegal_shared = set_a.intersection(set_b) - permitted
            
            assert not illegal_shared, (
                f"LỖI KIẾN TRÚC TỰ KIỂM CHỨNG (INVARIANT FAILURE): "
                f"'{target_a}' và '{target_b}' bị nhiễm chéo các nhãn không được phép: {illegal_shared}."
            )
            
    for key, aliases in CLASS_SYNONYMS_MAP.items():
        assert isinstance(aliases, list)
        for alias in aliases:
            assert isinstance(alias, str)
            assert alias == alias.lower().strip()

def _verify_no_taxonomy_overlap():
    """Đảm bảo mọi key trong CLASS_SYNONYMS_MAP đều là trường hợp thật sự
    cần alias — nếu key đã trùng thẳng tên 1 class trong taxonomy Tầng 1
    VÀ alias list của nó chỉ có đúng 1 phần tử (chính nó), nghĩa là entry 
    này thừa, nên xoá khỏi map tay để đơn giản hoá."""
    if not TAXONOMY_CLASSES_SET:
        return
        
    for key, aliases in list(CLASS_SYNONYMS_MAP.items()):
        if key in TAXONOMY_CLASSES_SET and len(aliases) == 1 and aliases[0] == key:
            logger.warning(f"REDUNDANT_ALIAS_ENTRY: '{key}' trùng thẳng taxonomy, không cần khai trong map tay.")

# Chạy hàm kiểm chứng ngay lúc load module.
_verify_invariants()
_verify_no_taxonomy_overlap()
logger.info("✅ CLASS_SYNONYMS_MAP Invariants Passed.")
