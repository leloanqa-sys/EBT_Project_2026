import os
import csv
from typing import List, Optional, Union
from src.common.schemas import SubmissionItem, SubmissionOutput, QueryType


def build_submission(
    ranked_items: List[SubmissionItem],
    query_id: str,
    query_type: Union[str, QueryType] = QueryType.KIS
) -> SubmissionOutput:
    """
    Constructs a SubmissionOutput object from ranked items.
    """
    if isinstance(query_type, str):
        try:
            q_type = QueryType(query_type)
        except ValueError:
            q_type = QueryType.KIS
    else:
        q_type = query_type

    return SubmissionOutput(
        query_id=query_id,
        query_type=q_type,
        items=ranked_items
    )

def format_submission(submission: SubmissionOutput, out_dir: str = "outputs") -> str:
    """
    Exports submission results into CSV format required for contest evaluation.
    Supports KIS, QA, and TRAKE query types.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_filepath = os.path.join(out_dir, f"submission_{submission.query_id}.csv")
    
    export_submission_csv(submission, out_filepath)
    return out_filepath

def export_submission_csv(output: SubmissionOutput, output_filepath: str) -> str:
    """
    Role C Output Formatter:
    Exports response candidates into CSV file format conforming to AIC 2026 standards.
    - KIS: query_id,rank,video_id,frame_id,confidence_score
    - QA: query_id,rank,video_id,frame_id,answer,confidence_score
    - TRAKE: query_id,rank,video_id,frame_id,confidence_score
    """
    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if output.query_type == QueryType.QA:
            writer.writerow(['query_id', 'rank', 'video_id', 'frame_id', 'answer', 'confidence_score'])
            for item in output.items:
                ans_str = item.answer or item.vqa_answer or ''
                writer.writerow([
                    output.query_id,
                    item.rank,
                    item.video_id,
                    item.frame_id,
                    ans_str,
                    f"{item.confidence_score:.6f}"
                ])
        else:
            writer.writerow(['query_id', 'rank', 'video_id', 'frame_id', 'confidence_score'])
            for item in output.items:
                writer.writerow([
                    output.query_id,
                    item.rank,
                    item.video_id,
                    item.frame_id,
                    f"{item.confidence_score:.6f}"
                ])

    return output_filepath


