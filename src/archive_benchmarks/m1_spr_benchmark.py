import os
import json
import time
from src.common.schemas import Query, VisualIRGraph
from src.role_b_nlp.query_parser import parse_query

TEST_CASES = [
    # TYPE 1: MULTI-REFERENCE CHAOS
    {
        "text": "Cô gái mặc váy hoa đang cho chim bồ câu ăn trước nhà thờ đá cổ kính.",
        "type": "Multi-ref",
        "expected_entities": ["girl", "dress", "bird", "church"],
        "expected_attributes": ["stone"],
        "expected_relations": ["front"]
    },
    {
        "text": "Vụ tai nạn giữa xe tải chở gỗ và xe khách màu đỏ ngay khúc cua đèo mù sương.",
        "type": "Multi-ref",
        "expected_entities": ["truck", "wood", "bus", "mountain", "fog"],
        "expected_attributes": ["red"],
        "expected_events": ["crash"]
    },
    {
        "text": "Chú mèo nhảy từ nóc tủ lạnh xuống bàn ăn làm đổ ly nước cam.",
        "type": "Multi-ref",
        "expected_entities": ["cat", "refrigerator", "table", "glass", "juice"],
        "expected_events": ["jump", "spill"]
    },
    {
        "text": "Một nhóm thanh niên trượt ván trên bậc thềm quảng trường trong lúc có người đang kéo đàn violin.",
        "type": "Multi-ref",
        "expected_entities": ["person", "skateboard", "stairs", "square", "violin"],
        "expected_events": ["skateboard", "play"]
    },
    {
        "text": "Chiếc trực thăng lượn vòng trên vùng biển xanh thẳm gần bãi san hô ngầm.",
        "type": "Multi-ref",
        "expected_entities": ["helicopter", "sea", "coral"],
        "expected_attributes": ["blue"],
        "expected_relations": ["above", "near"]
    },
    {
        "text": "Cầu thủ số 10 lừa bóng qua hai hậu vệ áo trắng rồi sút tung lưới góc cao bên trái.",
        "type": "Multi-ref",
        "expected_entities": ["player", "ball", "defender", "shirt", "net"],
        "expected_attributes": ["10", "white"],
        "expected_events": ["dribble", "shoot"]
    },
    {
        "text": "Người thợ lặn nhặt chiếc mỏ neo rỉ sét dưới đáy đại dương bên cạnh xác tàu đắm.",
        "type": "Multi-ref",
        "expected_entities": ["diver", "anchor", "ocean", "shipwreck"],
        "expected_attributes": ["rusty"],
        "expected_relations": ["under", "next_to"]
    },

    # TYPE 2: NEGATIVE / EXCLUSION
    {
        "text": "Bãi đậu xe trước siêu thị hoàn toàn trống không có chiếc xe nào.",
        "type": "Negative",
        "expected_entities": ["parking", "supermarket", "car"],
        "expected_relations": ["front"],
        "expected_polarity": ["NEGATIVE"]
    },
    {
        "text": "Đứa trẻ đứng khóc giữa trung tâm thương mại mà không có người lớn đi kèm.",
        "type": "Negative",
        "expected_entities": ["child", "mall", "adult"],
        "expected_events": ["cry"],
        "expected_polarity": ["NEGATIVE"]
    },
    {
        "text": "Chiếc xe đạp dựng ngoài cổng trường nhưng không bị khóa bánh.",
        "type": "Negative",
        "expected_entities": ["bicycle", "gate", "school", "wheel", "lock"],
        "expected_polarity": ["NEGATIVE"]
    },
    {
        "text": "Bàn làm việc lộn xộn giấy tờ nhưng không thấy chiếc máy tính xách tay.",
        "type": "Negative",
        "expected_entities": ["desk", "paper", "laptop"],
        "expected_polarity": ["NEGATIVE"]
    },
    {
        "text": "Trời mưa tầm tã nhưng chú chó nhỏ không tìm chỗ trú mưa.",
        "type": "Negative",
        "expected_entities": ["rain", "dog", "shelter"],
        "expected_polarity": ["NEGATIVE"]
    },
    {
        "text": "Căn phòng ngủ không có cửa sổ chỉ được thắp sáng bằng ngọn nến.",
        "type": "Negative",
        "expected_entities": ["bedroom", "window", "candle"],
        "expected_polarity": ["NEGATIVE"]
    },

    # TYPE 3: ABSTRACT / ACTION / TEMPORAL
    {
        "text": "Vận động viên thể dục dụng cụ hoàn thành cú santo trên không với tư thế tuyệt đẹp.",
        "type": "Abstract",
        "expected_entities": ["gymnast", "air"],
        "expected_events": ["somersault"]
    },
    {
        "text": "Hai người lính cứu hỏa đang nỗ lực dập tắt đám cháy lan nhanh từ tầng trệt lên lầu một.",
        "type": "Abstract",
        "expected_entities": ["firefighter", "fire", "floor"],
        "expected_events": ["extinguish", "spread"]
    },
    {
        "text": "Người đàn ông nhìn chằm chằm vào chiếc nhẫn cưới với ánh mắt đầy hối hận.",
        "type": "Abstract",
        "expected_entities": ["man", "ring"],
        "expected_events": ["stare"]
    },
    {
        "text": "Một con hẻm tối tăm mang lại cảm giác rùng rợn khi có tiếng bước chân vang vọng.",
        "type": "Abstract",
        "expected_entities": ["alley", "footstep"],
        "expected_attributes": ["dark"]
    },
    {
        "text": "Người bảo vệ già vừa ngáp ngủ vừa cầm đèn pin đi tuần tra lúc nửa đêm.",
        "type": "Abstract",
        "expected_entities": ["guard", "flashlight"],
        "expected_attributes": ["old"],
        "expected_events": ["yawn", "patrol"]
    },
    {
        "text": "Đôi bạn thân ôm chầm lấy nhau mừng rỡ sau nhiều năm xa cách tại sân bay.",
        "type": "Abstract",
        "expected_entities": ["friend", "airport"],
        "expected_events": ["hug"]
    },
    {
        "text": "Cảnh sát bắt giữ tên trộm đang cố gắng leo qua bức tường gạch phía sau ngôi nhà.",
        "type": "Abstract",
        "expected_entities": ["police", "thief", "wall", "brick", "house"],
        "expected_relations": ["behind"],
        "expected_events": ["arrest", "climb"]
    }
]

def main():
    os.makedirs("outputs", exist_ok=True)
    print(f"Running Stability SPR Benchmark on {len(TEST_CASES)} queries...")
    
    total_required = 0
    total_preserved = 0
    results_md = "# M1 Stability Benchmark Report\n\n"
    
    for idx, tc in enumerate(TEST_CASES):
        print(f"\n[{idx+1}/{len(TEST_CASES)}] [{tc['type']}] Query: '{tc['text']}'")
        q = Query(query_id=f"test_{idx}", raw_text=tc["text"], query_type="QA")
        
        # Sleep for rate limit safety
        time.sleep(1.2)
        try:
            ir_graph: VisualIRGraph = parse_query(q)
            
            # Print Graph structure visually
            graph_viz = "--- VISUAL IR GRAPH ---\n"
            for e in ir_graph.entities:
                graph_viz += f"Entity: {e.id} ({e.label})\n"
                for attr in ir_graph.attributes:
                    if attr.entity_id == e.id:
                        graph_viz += f"  └── Attribute: {attr.name} = {attr.value} [{attr.polarity.value}]\n"
                for rel in ir_graph.relations:
                    if rel.source_id == e.id:
                        graph_viz += f"  └── Relation: {rel.relation_type} -> {rel.target_id} [{rel.polarity.value}]\n"
            for ev in ir_graph.events:
                graph_viz += f"Event: {ev.action} ({ev.participants}) [{ev.polarity.value}]\n"
            for tc_time in ir_graph.temporal_constraints:
                graph_viz += f"Temporal: {tc_time.source_id} {tc_time.relation} {tc_time.target_id}\n"
            for oc in ir_graph.order_constraints:
                graph_viz += f"Order: {oc.target_id} {oc.axis} {oc.direction}\n"
            for sc in ir_graph.selection_constraints:
                graph_viz += f"Selection: {sc.target_id} rank {sc.rank}\n"
            graph_viz += "-----------------------\n"
            print(graph_viz)
            
            # Check Semantic Preservation
            preserved = 0
            required = 0
            
            # Entities
            json_dump = ir_graph.model_dump_json().lower()
            for ent in tc.get("expected_entities", []):
                required += 1
                if ent.lower() in json_dump:
                    preserved += 1
                    
            for attr in tc.get("expected_attributes", []):
                required += 1
                if attr.lower() in json_dump:
                    preserved += 1
                    
            for rel in tc.get("expected_relations", []):
                required += 1
                if rel.lower() in json_dump:
                    preserved += 1
                    
            for ev in tc.get("expected_events", []):
                required += 1
                if ev.lower() in json_dump:
                    preserved += 1
                    
            for pol in tc.get("expected_polarity", []):
                required += 1
                if pol.upper() in ir_graph.model_dump_json():
                    preserved += 1
                    
            print(f"SPR for query: {preserved}/{required}")
            total_required += required
            total_preserved += preserved
            
            results_md += f"### [{tc['type']}] {tc['text']}\n"
            results_md += f"**SPR:** {preserved}/{required}\n"
            results_md += "```text\n" + graph_viz + "```\n\n"
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            
    spr_percent = (total_preserved/total_required*100) if total_required > 0 else 0
    print(f"\nFinal Semantic Preservation Rate (SPR): {total_preserved}/{total_required} ({spr_percent:.1f}%)")
    results_md = f"## Final SPR: {total_preserved}/{total_required} ({spr_percent:.1f}%)\n\n" + results_md

    with open("outputs/m1_stability_benchmark.md", "w", encoding="utf-8") as f:
        f.write(results_md)

if __name__ == "__main__":
    main()
