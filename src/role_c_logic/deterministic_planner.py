from typing import List, Dict, Any
from dataclasses import dataclass, field
from src.common.schemas import VisualIRGraph, Polarity
from src.role_c_logic.capability_registry import registry, CapabilityStatus

@dataclass
class PlanStep:
    step_id: int
    operator_name: str
    target: str
    args: Dict[str, Any] = field(default_factory=dict)
    status: CapabilityStatus = CapabilityStatus.READY

@dataclass
class ExecutionPlan:
    query_id: str
    steps: List[PlanStep] = field(default_factory=list)
    has_unsupported: bool = False

def create_deterministic_plan(ir: VisualIRGraph) -> ExecutionPlan:
    """
    Biến VisualIRGraph thành ExecutionPlan tất định (DAG of Operators).
    Không dùng AI để lập kế hoạch.
    """
    plan = ExecutionPlan(query_id=ir.query_id)
    step_counter = 1
    
    # 1. RETRIEVAL (Luôn có để lấy base candidates)
    # G0/G2 Experiment Result: Use raw_text to preserve semantic structure and maintain Recall@500
    clip_text = ir.raw_text
        
    op_retrieve = registry.get_operator("CLIP_RETRIEVE")
    plan.steps.append(PlanStep(
        step_id=step_counter,
        operator_name="CLIP_RETRIEVE",
        target="video_frames",
        args={"text": clip_text, "k": 500},
        status=op_retrieve.status if op_retrieve else CapabilityStatus.UNSUPPORTED
    ))
    step_counter += 1
    
    # 2. DETECT ENTITIES
    for ent in ir.entities:
        op_detect = registry.get_operator("DETECT")
        plan.steps.append(PlanStep(
            step_id=step_counter,
            operator_name="DETECT",
            target=ent.id,
            args={"class": ent.label},
            status=op_detect.status if op_detect else CapabilityStatus.UNSUPPORTED
        ))
        step_counter += 1

    # 3. FILTER ATTRIBUTES
    for attr in ir.attributes:
        op_filter = registry.get_operator("FILTER_ATTRIBUTE")
        plan.steps.append(PlanStep(
            step_id=step_counter,
            operator_name="FILTER_ATTRIBUTE",
            target=attr.entity_id,
            args={"name": attr.name, "value": attr.value, "polarity": attr.polarity.value},
            status=op_filter.status if op_filter else CapabilityStatus.UNSUPPORTED
        ))
        if op_filter is None or op_filter.status == CapabilityStatus.UNSUPPORTED:
            plan.has_unsupported = True
        step_counter += 1

    # 4. SPATIAL CHECKS / RELATIONS
    for rel in ir.relations:
        spatial_op_name = registry.resolve_spatial_operator(rel.relation_type)
        if spatial_op_name:
            op_spatial = registry.get_operator(spatial_op_name)
            plan.steps.append(PlanStep(
                step_id=step_counter,
                operator_name=spatial_op_name,
                target=f"{rel.source_id}_{rel.target_id}",
                args={"source": rel.source_id, "target": rel.target_id, "polarity": rel.polarity.value},
                status=op_spatial.status if op_spatial else CapabilityStatus.UNSUPPORTED
            ))
            if op_spatial is None or op_spatial.status == CapabilityStatus.UNSUPPORTED:
                plan.has_unsupported = True
        else:
            # Not a spatial relation or not recognized
            plan.steps.append(PlanStep(
                step_id=step_counter,
                operator_name=f"RELATION_{rel.relation_type.upper()}",
                target=f"{rel.source_id}_{rel.target_id}",
                args={"source": rel.source_id, "target": rel.target_id, "polarity": rel.polarity.value},
                status=CapabilityStatus.UNSUPPORTED
            ))
            plan.has_unsupported = True
        step_counter += 1

    # 5. EVENTS (Abstract Actions)
    for ev in ir.events:
        op_event = registry.get_operator("EVENT_ACTION")
        if op_event and op_event.status in (CapabilityStatus.EXPERIMENTAL, CapabilityStatus.DEFER):
            # Mount to VLM_VERIFY Adapter
            plan.steps.append(PlanStep(
                step_id=step_counter,
                operator_name="VLM_VERIFY",
                target=ev.id,
                args={"action": ev.action, "participants": ev.participants, "polarity": ev.polarity.value, "reason": "action_recognition"},
                status=CapabilityStatus.READY
            ))
        else:
            plan.steps.append(PlanStep(
                step_id=step_counter,
                operator_name="EVENT_ACTION",
                target=ev.id,
                args={"action": ev.action, "participants": ev.participants, "polarity": ev.polarity.value},
                status=op_event.status if op_event else CapabilityStatus.UNSUPPORTED
            ))
            if op_event is None or op_event.status == CapabilityStatus.UNSUPPORTED:
                plan.has_unsupported = True
        step_counter += 1
        
    return plan
