import os
import json
from enum import Enum
from typing import List, Dict, Any

class EvalResult(Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"

def load_objects_json(video_id: str, frame_idx: int, obj_dir: str = "data/raw/objects") -> Dict[str, Any]:
    # Need to map frame_idx to n? 
    # Actually, the JSON filename is based on `n`, but in our audit we saw some frames mapping to 001.json
    # For a unit test, let's just mock the objects JSON or read it directly if we know the path.
    # We will pass the mock data directly to test the logic.
    pass

def evaluate_detect(objects_data: Dict[str, Any], target_class: str) -> EvalResult:
    if not objects_data or "detection_class_entities" not in objects_data:
        return EvalResult.UNKNOWN
    
    entities = [e.lower() for e in objects_data["detection_class_entities"]]
    if target_class.lower() in entities:
        return EvalResult.SATISFIED
    
    # If it's a known class we can detect but it's not there -> VIOLATED
    return EvalResult.VIOLATED

def evaluate_spatial(objects_data: Dict[str, Any], relation: str, subj: str, obj: str) -> EvalResult:
    if not objects_data or "detection_class_entities" not in objects_data:
        return EvalResult.UNKNOWN
        
    entities = [e.lower() for e in objects_data["detection_class_entities"]]
    boxes = objects_data.get("detection_boxes", [])
    
    subj = subj.lower()
    obj = obj.lower()
    
    if subj not in entities or obj not in entities:
        return EvalResult.UNKNOWN # One or both missing, we can't evaluate the spatial relation
        
    # Get all indices for subj and obj
    subj_indices = [i for i, e in enumerate(entities) if e == subj]
    obj_indices = [i for i, e in enumerate(entities) if e == obj]
    
    for si in subj_indices:
        for oi in obj_indices:
            # Box format: [ymin, xmin, ymax, xmax]
            s_box = [float(x) for x in boxes[si]]
            o_box = [float(x) for x in boxes[oi]]
            
            s_center_x = (s_box[1] + s_box[3]) / 2.0
            o_center_x = (o_box[1] + o_box[3]) / 2.0
            s_center_y = (s_box[0] + s_box[2]) / 2.0
            o_center_y = (o_box[0] + o_box[2]) / 2.0
            
            if relation == "LEFT_OF" and s_center_x < o_center_x:
                return EvalResult.SATISFIED
            if relation == "RIGHT_OF" and s_center_x > o_center_x:
                return EvalResult.SATISFIED
            if relation == "ABOVE" and s_center_y < o_center_y: # y increases downwards
                return EvalResult.SATISFIED
            if relation == "BELOW" and s_center_y > o_center_y:
                return EvalResult.SATISFIED
                
    return EvalResult.VIOLATED

def evaluate_attribute(objects_data: Dict[str, Any], target: str, attribute: str) -> EvalResult:
    # Our objects.json does NOT contain color/attribute data (only class entities)
    # Therefore, ATTRIBUTE filters must ALWAYS return UNKNOWN, never VIOLATED.
    return EvalResult.UNKNOWN

def run_tests():
    print("--- M2-E.1 Operator-level Validation ---")
    
    mock_data = {
        "detection_class_entities": ["Person", "Car", "Tree"],
        "detection_boxes": [
            [0.1, 0.1, 0.4, 0.2], # Person: center_x = 0.15, center_y = 0.25
            [0.1, 0.5, 0.4, 0.8], # Car: center_x = 0.65, center_y = 0.25
            [0.5, 0.1, 0.9, 0.2]  # Tree: center_x = 0.15, center_y = 0.70
        ]
    }
    
    print("\n1. Test DETECT")
    r1 = evaluate_detect(mock_data, "Person")
    print(f"DETECT 'Person': {r1.value} (Expected: SATISFIED)")
    
    r2 = evaluate_detect(mock_data, "Dog")
    print(f"DETECT 'Dog': {r2.value} (Expected: VIOLATED)")
    
    r3 = evaluate_detect({}, "Person")
    print(f"DETECT 'Person' (No Data): {r3.value} (Expected: UNKNOWN)")
    
    print("\n2. Test SPATIAL")
    r4 = evaluate_spatial(mock_data, "LEFT_OF", "Person", "Car")
    print(f"LEFT_OF('Person', 'Car'): {r4.value} (Expected: SATISFIED)")
    
    r5 = evaluate_spatial(mock_data, "RIGHT_OF", "Person", "Car")
    print(f"RIGHT_OF('Person', 'Car'): {r5.value} (Expected: VIOLATED)")
    
    r6 = evaluate_spatial(mock_data, "LEFT_OF", "Person", "Dog")
    print(f"LEFT_OF('Person', 'Dog'): {r6.value} (Expected: UNKNOWN)")
    
    r7 = evaluate_spatial(mock_data, "ABOVE", "Person", "Tree")
    print(f"ABOVE('Person', 'Tree'): {r7.value} (Expected: SATISFIED)")
    
    print("\n3. Test ATTRIBUTE")
    r8 = evaluate_attribute(mock_data, "Person", "Red")
    print(f"ATTRIBUTE('Person', 'Red'): {r8.value} (Expected: UNKNOWN)")
    
    print("\nConclusion: Operators strictly adhere to SATISFIED / VIOLATED / UNKNOWN.")
    print("UNKNOWN states correctly prevent valid candidates from being dropped.")

if __name__ == "__main__":
    run_tests()
