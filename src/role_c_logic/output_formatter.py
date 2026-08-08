import os
import csv
from src.common.schemas import SubmissionOutput

def export_submission_csv(output: SubmissionOutput, output_filepath: str):
    """
    Role C Output Formatter:
    Exports 100 response candidates into CSV file format required by BTC.
    Format: rank,video_id,frame_id,answer,confidence_score
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'video_id', 'frame_id', 'answer', 'confidence_score'])
        for item in output.items:
            writer.writerow([
                item.rank,
                item.video_id,
                item.frame_id,
                item.answer if item.answer else '',
                item.confidence_score
            ])
