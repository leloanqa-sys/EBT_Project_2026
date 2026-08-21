"""
MVP Demo - Interactive CLI for testing the full pipeline.
Usage:
  python -m src.mvp_demo
  python -m src.mvp_demo --query "người đàn ông áo đỏ đứng phát biểu" --type KIS
  python -m src.mvp_demo --batch data/m2_10_human_gt.json
"""
import argparse
import json
import sys
import os

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.pipeline import MVPPipeline, to_submission, export_csv, print_trace_summary


def run_interactive(pipeline: MVPPipeline):
    print("\n=== EBT MVP Interactive Demo ===")
    print("Nhập query bằng tiếng Việt hoặc tiếng Anh. Gõ 'quit' để thoát.\n")
    qid_counter = 1
    while True:
        try:
            query = input(f"[Q{qid_counter:03d}] Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query or query.lower() == "quit":
            break

        qtype = input("  Type [KIS/QA/TRAKE, default KIS]: ").strip().upper() or "KIS"
        qid = f"DEMO_{qid_counter:03d}"

        print(f"\n  Processing '{query[:70]}'...")
        result = pipeline.run(qid, query, qtype)

        print(f"\n  ✅ Done in {result.latency_ms:.0f}ms | {len(result.candidates)} candidates returned")
        print_trace_summary(result)

        if result.candidates:
            print(f"\n  Top-5 Results:")
            for i, c in enumerate(result.candidates[:5], 1):
                print(f"    #{i:2d}  {c.video_id}  frame={c.frame_idx:4d}  score={c.siglip_score:.4f}")

        sub = to_submission(result)
        out_path = f"outputs/submission_{qid}.csv"
        export_csv([sub], out_path)
        print()
        qid_counter += 1


def run_batch(pipeline: MVPPipeline, gt_file: str):
    print(f"\n=== EBT MVP Batch Mode: {gt_file} ===")
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    queries = [{"id": q["query_id"], "text": q["query"], "type": q.get("type", "KIS")} for q in gt_data]
    results = pipeline.run_batch(queries)

    # Evaluate if GT available
    hits_at_500 = 0
    hits_at_5 = 0
    for result, q in zip(results, gt_data):
        gt_vid = q.get("gt", {}).get("video_id", "")
        retrieved_vids = [c.video_id for c in result.candidates]
        if gt_vid in retrieved_vids:
            hits_at_500 += 1
        if gt_vid in retrieved_vids[:5]:
            hits_at_5 += 1

    total = len(results)
    print(f"\n=== Batch Results ===")
    print(f"  Queries: {total}")
    print(f"  Recall@500: {hits_at_500/total*100:.1f}%")
    print(f"  P@5:        {hits_at_5/total*100:.1f}%")

    subs = [to_submission(r) for r in results]
    out_path = "outputs/submission_mvp_batch.csv"
    export_csv(subs, out_path)
    print(f"  Submission: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EBT MVP Pipeline Demo")
    parser.add_argument("--query", type=str, help="Single query text")
    parser.add_argument("--type", type=str, default="KIS", help="Query type: KIS/QA/TRAKE")
    parser.add_argument("--batch", type=str, help="Path to GT JSON file for batch evaluation")
    parser.add_argument("--threshold", type=float, default=0.3, help="DETECT score threshold (default: 0.3)")
    args = parser.parse_args()

    pipeline = MVPPipeline(detect_threshold=args.threshold)

    if args.batch:
        run_batch(pipeline, args.batch)
    elif args.query:
        pipeline._load()
        result = pipeline.run("CLI_001", args.query, args.type)
        print(f"\n  Done in {result.latency_ms:.0f}ms | {len(result.candidates)} candidates")
        print_trace_summary(result)
        if result.candidates:
            print("\n  Top-10:")
            for i, c in enumerate(result.candidates[:10], 1):
                print(f"    #{i:2d}  {c.video_id}  frame={c.frame_idx:4d}  score={c.siglip_score:.4f}")
        sub = to_submission(result)
        export_csv([sub], "outputs/submission_cli.csv")
    else:
        pipeline._load()
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
