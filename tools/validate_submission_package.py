import os
import sys
import glob
import re
import csv
import zipfile
from pathlib import Path

def validate_package(zip_path="submission.zip", queries_dir="data/contest_queries"):
    print("=" * 70)
    print(f"VALIDATING SUBMISSION PACKAGE: {zip_path}")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    if not os.path.exists(zip_path):
        print(f"[FAIL] Zip file {zip_path} does not exist!")
        return False
        
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        
    print(f"Total files in zip: {len(namelist)}")
    
    # Check 1: All files in submission/ folder
    non_sub = [f for f in namelist if not f.startswith("submission/") or f == "submission/"]
    if non_sub:
        errors.append(f"Found items not in 'submission/' prefix: {non_sub}")
        
    csv_in_zip = [f for f in namelist if f.startswith("submission/") and f.endswith(".csv")]
    print(f"CSV files in submission/: {len(csv_in_zip)}")
    
    # Check 2: Match with queries_dir
    expected_qids = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(queries_dir, "*.txt"))]
    print(f"Expected queries count: {len(expected_qids)}")
    
    found_qids = [os.path.splitext(os.path.basename(f))[0] for f in csv_in_zip]
    missing_qids = set(expected_qids) - set(found_qids)
    extra_qids = set(found_qids) - set(expected_qids)
    
    if missing_qids:
        errors.append(f"Missing CSV files for queries: {missing_qids}")
    if extra_qids:
        warnings.append(f"Extra CSV files found: {extra_qids}")
        
    # Check 3: Validate each CSV content
    extract_temp_dir = "temp_validation_check"
    os.makedirs(extract_temp_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_temp_dir)
        
    for qid in expected_qids:
        csv_file = os.path.join(extract_temp_dir, "submission", f"{qid}.csv")
        if not os.path.exists(csv_file):
            continue
            
        with open(csv_file, "r", encoding="utf-8", errors="strict") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        if len(lines) == 0:
            errors.append(f"[{qid}.csv] File is empty!")
            continue
        if len(lines) > 100:
            errors.append(f"[{qid}.csv] Row count exceeds 100 limit ({len(lines)} rows)!")
            
        # Parse CSV
        reader = csv.reader(lines, delimiter=",")
        parsed_rows = list(reader)
        
        # Check first line for header row violation
        first_row = parsed_rows[0]
        if first_row[0].lower() in ["video_id", "query_id", "rank", "video"]:
            errors.append(f"[{qid}.csv] CSV contains a HEADER ROW on line 1: {first_row}")
            
        # Specific check per query type
        if qid.endswith("-kis") or "-kis" in qid:
            for row_idx, r in enumerate(parsed_rows, 1):
                if len(r) != 2:
                    errors.append(f"[{qid}.csv:L{row_idx}] KIS row must have exactly 2 columns, got {len(r)}: {r}")
                    break
                vid, fidx = r[0].strip(), r[1].strip()
                if vid.endswith(".mp4"):
                    errors.append(f"[{qid}.csv:L{row_idx}] Video ID contains .mp4 extension: '{vid}'")
                if not fidx.isdigit():
                    errors.append(f"[{qid}.csv:L{row_idx}] Frame ID is not a valid integer: '{fidx}'")
                    
        elif qid.endswith("-qa") or "-qa" in qid:
            for row_idx, r in enumerate(parsed_rows, 1):
                if len(r) < 3:
                    errors.append(f"[{qid}.csv:L{row_idx}] QA row must have at least 3 columns, got {len(r)}: {r}")
                    break
                vid, fidx = r[0].strip(), r[1].strip()
                ans = ",".join(r[2:]).strip()
                if vid.endswith(".mp4"):
                    errors.append(f"[{qid}.csv:L{row_idx}] Video ID contains .mp4 extension: '{vid}'")
                if not fidx.isdigit():
                    errors.append(f"[{qid}.csv:L{row_idx}] Frame ID is not a valid integer: '{fidx}'")
                if len(ans) > 100:
                    errors.append(f"[{qid}.csv:L{row_idx}] Answer length > 100 chars ({len(ans)} chars): '{ans}'")
                    
        elif qid.endswith("-trake") or "-trake" in qid:
            # Expected N events: 4
            for row_idx, r in enumerate(parsed_rows, 1):
                if len(r) < 3:
                    errors.append(f"[{qid}.csv:L{row_idx}] TRAKE row has too few columns ({len(r)}): {r}")
                    break
                vid = r[0].strip()
                if vid.endswith(".mp4"):
                    errors.append(f"[{qid}.csv:L{row_idx}] Video ID contains .mp4 extension: '{vid}'")
                f_ints = []
                for val in r[1:]:
                    if not val.strip().isdigit():
                        errors.append(f"[{qid}.csv:L{row_idx}] TRAKE frame ID is not integer: '{val}'")
                    else:
                        f_ints.append(int(val.strip()))
                # Check chronological order
                for i in range(1, len(f_ints)):
                    if f_ints[i] <= f_ints[i-1]:
                        warnings.append(f"[{qid}.csv:L{row_idx}] Non-strictly increasing frame sequence: {f_ints}")
                        break
                        
    # Clean up temp
    import shutil
    shutil.rmtree(extract_temp_dir, ignore_errors=True)
    
    print("-" * 70)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  [WARN] {w}")
    if errors:
        print(f"FAILURES / ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  [ERROR] {e}")
        print(f"\n[RESULT] VALIDATION FAILED with {len(errors)} errors.")
        return False
    else:
        print("\n[RESULT] ALL CHECKS PASSED PERFECTLY! 100% CONTEST-COMPLIANT.")
        return True

if __name__ == '__main__':
    validate_package()
