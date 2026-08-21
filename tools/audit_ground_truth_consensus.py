"""Audit conflicting ground-truth sources and submission evidence.

The report is diagnostic only. It never changes a ground-truth file or a
submission package and should be run before using any source for tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
GT_FILES = (
    ROOT / "data" / "official_preliminary_gt.json",
    ROOT / "data" / "ground_truth_isolated_clean.json",
    ROOT / "data" / "preliminary_24_ground_truth.json",
)


def load_gt(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["query_id"]: item for item in data}


def target_from_gt(item: dict[str, Any]) -> tuple[str, int | None, str]:
    gt = item.get("gt", {})
    video = str(gt.get("video_id", gt.get("target_video", ""))).strip()
    frame = gt.get("keyframe", gt.get("frame_start"))
    try:
        frame = int(frame) if frame is not None else None
    except (TypeError, ValueError):
        frame = None
    answer = str(gt.get("answer", "")).strip()
    return video, frame, answer


def first_submission_row(path: Path, query_id: str) -> tuple[str, int | None, str]:
    with zipfile.ZipFile(path) as package:
        name = f"submission/{query_id}.csv"
        if name not in package.namelist():
            return "", None, ""
        rows = list(csv.reader(package.read(name).decode("utf-8-sig", errors="replace").splitlines()))
    if not rows:
        return "", None, ""
    row = rows[0]
    video = row[0].strip() if row else ""
    try:
        frame = int(row[1]) if len(row) > 1 else None
    except ValueError:
        frame = None
    answer = ",".join(row[2:]).strip().strip('"') if len(row) > 2 else ""
    return video, frame, answer


def frame_exists(conn: sqlite3.Connection, video: str, frame: int | None) -> bool:
    if not video or frame is None:
        return False
    return conn.execute(
        "SELECT 1 FROM keyframes WHERE video_id=? AND frame_idx=? LIMIT 1",
        (video, frame),
    ).fetchone() is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ground_truth_consensus_audit.json")
    args = parser.parse_args()

    gt_sources = {path.name: load_gt(path) for path in GT_FILES if path.exists()}
    packages = {path.name: path for path in ROOT.glob("submission_*.zip")}
    old = next((path for path in ROOT.rglob("submission.zip") if path.parent != ROOT), None)
    if old is not None:
        packages["old_5_2_submission.zip"] = old

    query_ids = sorted({qid for source in gt_sources.values() for qid in source})
    conn = sqlite3.connect(ROOT / "data" / "processed" / "media.db")
    report: dict[str, Any] = {"ground_truth_sources": sorted(gt_sources), "packages": sorted(packages), "queries": {}}

    for query_id in query_ids:
        gt_rows = {}
        gt_votes = Counter()
        for source_name, source in gt_sources.items():
            if query_id not in source:
                continue
            video, frame, answer = target_from_gt(source[query_id])
            key = (video, frame)
            gt_votes[key] += 1
            gt_rows[source_name] = {
                "type": source[query_id].get("type", ""),
                "video_id": video,
                "frame": frame,
                "answer": answer,
                "frame_exists": frame_exists(conn, video, frame),
            }

        package_rows = {}
        package_votes = Counter()
        for package_name, package_path in packages.items():
            video, frame, answer = first_submission_row(package_path, query_id)
            package_votes[(video, frame)] += 1
            package_rows[package_name] = {
                "video_id": video,
                "frame": frame,
                "answer": answer,
                "frame_exists": frame_exists(conn, video, frame),
            }

        gt_unique = [key for key in gt_votes if key != ("", None)]
        status = "AGREED" if len(gt_unique) == 1 else ("MISSING_TARGET" if not gt_unique else "CONFLICT")
        old_submission = package_rows.get("old_5_2_submission.zip", {})
        old_key = (old_submission.get("video_id", ""), old_submission.get("frame"))
        old_matches_gt = old_key in gt_votes and old_key != ("", None)
        if status == "AGREED" and old_matches_gt:
            review_status = "SAFE_FOR_OFFLINE_TUNING"
        elif status == "AGREED":
            review_status = "GT_AGREED_BUT_PACKAGE_DIFFERS"
        else:
            review_status = "MANUAL_VISUAL_REVIEW_REQUIRED"

        report["queries"][query_id] = {
            "ground_truth": gt_rows,
            "ground_truth_votes": {f"{video}:{frame}": count for (video, frame), count in gt_votes.items()},
            "submission_evidence": package_rows,
            "submission_votes": {f"{video}:{frame}": count for (video, frame), count in package_votes.items()},
            "status": status,
            "old_5_2_matches_any_gt": old_matches_gt,
            "review_status": review_status,
        }

    conn.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(item["review_status"] for item in report["queries"].values())
    print(f"Wrote {args.output}")
    print("status_counts", dict(counts))
    print("packages", ", ".join(packages))


if __name__ == "__main__":
    main()