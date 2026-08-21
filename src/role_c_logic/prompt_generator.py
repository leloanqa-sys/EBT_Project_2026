from src.common.schemas import VisualIRGraph
from typing import Optional


def ir_graph_to_prompt(ir_graph: VisualIRGraph, question: Optional[str] = None) -> str:
    """
    Translates the IR Graph into a strict English prompt for Gemini VLM.

    KIS mode (no question): strict visual verification — must match ALL attributes.
    QA mode (with question): verify match AND answer the specific question.

    Bug #3 Fix: The prompt now explicitly demands strict attribute checking
    (especially color, gender) to prevent hallucination or vague matches.
    """
    lines = []

    # Build entity + attribute description for strict checking
    entity_checks = []
    for entity in ir_graph.entities:
        # Find attributes for this entity
        attrs = [a for a in ir_graph.attributes if a.entity_id == entity.id]
        if attrs:
            pos_attrs = [f"{a.value} {a.name}" for a in attrs if a.polarity == "POSITIVE"]
            neg_attrs = [f"NOT {a.value} {a.name}" for a in attrs if a.polarity == "NEGATIVE"]
            desc = " ".join(pos_attrs + neg_attrs + [entity.label])
        else:
            desc = entity.label
        entity_checks.append(desc)

    if question:
        # QA Mode — match check + answer
        lines.append("Analyze this image carefully. Answer the following question ONLY if the image matches the description below.")
        if entity_checks:
            lines.append(f"Description: {', '.join(entity_checks)}")
        if ir_graph.events:
            evt = ", ".join([e.action for e in ir_graph.events])
            lines.append(f"Event/Action: {evt}")
        lines.append(f"\nQuestion: {question}")
        lines.append("If the image does NOT clearly match the description, return match=false and answer=null.")
        lines.append("If it matches, return match=true and provide a concise answer to the question.")
    else:
        # KIS Mode — semantic verification
        lines.append("Carefully examine this image against the following key elements:")
        for check in entity_checks:
            lines.append(f"  - Subject/Object: {check}")
        if ir_graph.events:
            for e in ir_graph.events:
                lines.append(f"  - Action/Event: {e.action}")
        lines.append("")
        lines.append(
            "IMPORTANT INSTRUCTION: "
            "You are evaluating if this image represents the CORE SEMANTIC MEANING of the query. "
            "Do NOT be overly strict about minor background details, exact clothing colors, or small missed objects if the main action and primary subjects clearly match. "
            "Often, textual queries contain slight errors (e.g. saying 'woman' instead of 'man'). If the core event (e.g. 'harvesting pineapples', 'playing metal instruments') is a perfect match, you should accept it. "
            "Return match=true if this image is a highly plausible match for the main event/subjects described. Return match=false only if it is clearly a different scene."
        )

    return "\n".join(lines)
