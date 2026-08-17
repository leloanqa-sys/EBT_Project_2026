# [Goal Description]
Fix the Q&A query processing and CSV export logic so that it conforms to the AI Challenge 2026 (AIC 2026) requirements and produces accurate QA verdicts. The current system fails entirely on QA queries due to frontend routing bugs, missing question extraction, and incorrect CSV formatting.

## User Review Required
The changes will consolidate QA and KIS routing into `kis_routes.py` so the UI doesn't need to split `query` and `question` explicitly. Is this acceptable, or do you prefer to keep `/search/qa` and add a new input box in the UI?
> [!IMPORTANT]
> The current plan modifies `kis_routes.py` to auto-handle QA queries seamlessly so the user only needs one search box.

## Open Questions
None.

## Proposed Changes

### 1. Frontend (`api/static/app.js`)
#### [MODIFY] [app.js](file:///c:/Users/Admin/Downloads/EBT_Project_2026/api/static/app.js)
- Fix the CSV export logic for QA. The `data.results` list doesn't always have `vqa_answer` for every frame (since we only escalate top 10 to VLM). We will grab the first available `vqa_answer` and broadcast it to all 100 rows in the CSV.
- Ensure `query_type` comparison is robust (e.g., `data.query_type === 'QA' || data.query_type === 'Q&A'`).

### 2. Backend Routing (`api/routes/kis_routes.py`)
#### [MODIFY] [kis_routes.py](file:///c:/Users/Admin/Downloads/EBT_Project_2026/api/routes/kis_routes.py)
- When receiving a request with `query_type == "QA"`, if `req.question` is None, we set `question = req.query` so that `pipeline.py` triggers the VLM answering logic.

### 3. Pipeline (`src/pipeline.py`)
#### [MODIFY] [pipeline.py](file:///c:/Users/Admin/Downloads/EBT_Project_2026/src/pipeline.py)
- In `MVPPipeline.run`, if `query_type == "QA"` and `question` is provided, ensure we use it to prompt the VLM. The current code uses `prompt_version = "v2_qa" if question else "v1"`. We will ensure the VLM response parses the `answer` field and assigns it to `c.vqa_answer`.

## Verification Plan

### Automated Tests
- N/A

### Manual Verification
- Start the server, enter a Q&A query ("trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì") with the QA tab selected.
- Verify the UI displays the results and correctly hits the VLM endpoint.
- Click "Xuất CSV" and check if the exported CSV has `video_id,frame_idx,answer` columns and contains the extracted answer.
