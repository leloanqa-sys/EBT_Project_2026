from enum import Enum
from typing import List, Dict, Any, Optional

class CapabilityStatus(str, Enum):
    READY = "READY"
    EXPERIMENTAL = "EXPERIMENTAL"
    UNSUPPORTED = "UNSUPPORTED"
    DEFER = "DEFER"

class OperatorCost(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class OperatorDef:
    def __init__(self, name: str, executor: str, cost: OperatorCost, status: CapabilityStatus):
        self.name = name
        self.executor = executor
        self.cost = cost
        self.status = status

class CapabilityRegistry:
    """
    Registry mô tả những năng lực MÀ HỆ THỐNG THỰC SỰ ĐANG CÓ.
    Không mô tả những năng lực ước ao có.
    """
    def __init__(self):
        self.operators: Dict[str, OperatorDef] = {
            # Core Retrieval & Detection
            "CLIP_RETRIEVE": OperatorDef("CLIP_RETRIEVE", "FAISS + CLIP", OperatorCost.LOW, CapabilityStatus.READY),
            "DETECT": OperatorDef("DETECT", "Faster R-CNN", OperatorCost.LOW, CapabilityStatus.READY),
            
            # Attribute Filters
            # M0 showed that attribute/color filtering is a bottleneck, so it's marked EXPERIMENTAL
            "FILTER_ATTRIBUTE": OperatorDef("FILTER_ATTRIBUTE", "CLIP/Crop", OperatorCost.MEDIUM, CapabilityStatus.EXPERIMENTAL),
            
            # Spatial Checks
            "SPATIAL_LEFT_OF": OperatorDef("SPATIAL_LEFT_OF", "BBox Geometry", OperatorCost.VERY_LOW, CapabilityStatus.READY),
            "SPATIAL_RIGHT_OF": OperatorDef("SPATIAL_RIGHT_OF", "BBox Geometry", OperatorCost.VERY_LOW, CapabilityStatus.READY),
            "SPATIAL_ABOVE": OperatorDef("SPATIAL_ABOVE", "BBox Geometry", OperatorCost.VERY_LOW, CapabilityStatus.READY),
            "SPATIAL_BELOW": OperatorDef("SPATIAL_BELOW", "BBox Geometry", OperatorCost.VERY_LOW, CapabilityStatus.READY),
            # "Behind" is hard to infer purely from 2D bounding boxes without depth estimation, so EXPERIMENTAL
            "SPATIAL_BEHIND": OperatorDef("SPATIAL_BEHIND", "BBox Geometry", OperatorCost.VERY_LOW, CapabilityStatus.EXPERIMENTAL),
            "SPATIAL_FRONT": OperatorDef("SPATIAL_FRONT", "BBox Geometry", OperatorCost.VERY_LOW, CapabilityStatus.EXPERIMENTAL),
            
            # Advanced / Missing Capabilities -> Deferred to VQA (Qwen2-VL)
            "COUNT": OperatorDef("COUNT", "None", OperatorCost.HIGH, CapabilityStatus.DEFER),
            "OCR": OperatorDef("OCR", "None", OperatorCost.HIGH, CapabilityStatus.DEFER),
            "EVENT_ACTION": OperatorDef("EVENT_ACTION", "Action Recognition", OperatorCost.HIGH, CapabilityStatus.DEFER),
            "VLM_VERIFY": OperatorDef("VLM_VERIFY", "Gemini Vision", OperatorCost.HIGH, CapabilityStatus.DEFER)
        }
    
    def get_operator(self, name: str) -> Optional[OperatorDef]:
        return self.operators.get(name)

    def resolve_spatial_operator(self, relation: str) -> str:
        """Maps a natural language relation to an exact Spatial Operator."""
        mapping = {
            "left_of": "SPATIAL_LEFT_OF",
            "right_of": "SPATIAL_RIGHT_OF",
            "above": "SPATIAL_ABOVE",
            "below": "SPATIAL_BELOW",
            "behind": "SPATIAL_BEHIND",
            "in_front_of": "SPATIAL_FRONT",
            "front": "SPATIAL_FRONT"
        }
        return mapping.get(relation.lower())

registry = CapabilityRegistry()
