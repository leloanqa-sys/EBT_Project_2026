"""Build an ensemble-retrieval candidate without changing the locked baseline."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schemas import QueryType
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_b_nlp.gemini_nlp_engine import compile_to_visual_ir
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor


REFERENCE_ZIP = PROJECT_ROOT / "dữ liệu cũ" / "submission.zip"
OUTPUT_ZIP = PROJECT_ROOT / "outputs" / "submission_ensemble_kis.zip"


def fenced_rows(conn: sqlite3.Connection, video_id: str, peak_frame: int, count: int) -> list[int]:
    rows = conn.execute(
        "SELECT frame_idx FROM keyframes WHERE video_id=? ORDER BY ABS(frame_idx-?) LIMIT ?",
        (video_id, peak_frame, count),
    ).fetchall()
    return [int(row[0]) for row in rows] or [peak_frame]


def build_kis_rows(
    searcher: VectorSearcher,
    executor: DeterministicExecutor,
    conn: sqlite3.Connection,
    query_id: str,
    query_text: str,
) -> list[str]:
    from src.pipeline import MVPPipeline, to_submission
    
    pipeline = MVPPipeline(searcher=searcher)
    pipeline._executor = executor
    
    result = pipeline.run(query_id, query_text, "KIS")
    submission = to_submission(result, top_k=100)
    
    rows = []
    for item in submission.items:
        rows.append(f"{item.video_id},{item.frame_id}")
    return rows


def main() -> None:
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    searcher = VectorSearcher()
    executor = DeterministicExecutor(searcher)
    conn = sqlite3.connect(PROJECT_ROOT / "data" / "processed" / "media.db")

    with zipfile.ZipFile(REFERENCE_ZIP, "r") as source, zipfile.ZipFile(
        OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for name in source.namelist():
            if not name.endswith(".csv"):
                continue
            query_id = Path(name).stem
            if query_id.endswith("-kis"):
                query_path = PROJECT_ROOT / "data" / "contest_queries" / f"{query_id}.txt"
                query_text = query_path.read_text(encoding="utf-8", errors="replace").strip()
                rows = build_kis_rows(searcher, executor, conn, query_id, query_text)
                if not rows:
                    raise RuntimeError(f"No ensemble candidates for {query_id}")
                payload = "\r\n".join(rows) + "\r\n"
                target.writestr(name, payload.encode("utf-8"))
            else:
                target.writestr(name, source.read(name))

    conn.close()
    print(f"Wrote ensemble candidate: {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()