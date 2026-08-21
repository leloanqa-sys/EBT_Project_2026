import os
import re
import urllib.parse
from pathlib import Path

def main():
    queries_dir = Path("data/contest_queries")
    output_html = Path("api/static/exam.html")
    
    if not queries_dir.exists():
        print(f"Error: {queries_dir} does not exist.")
        return

    # Find and parse files
    pattern = re.compile(r"query-p1-(\d+)-(\w+)\.txt")
    exam_items = []
    
    for f in queries_dir.iterdir():
        if f.is_file() and f.suffix == ".txt":
            match = pattern.match(f.name)
            if match:
                q_num = int(match.group(1))
                q_type = match.group(2).upper()
                
                with open(f, "r", encoding="utf-8") as file_handle:
                    q_text = file_handle.read().strip()
                
                exam_items.append({
                    "number": q_num,
                    "type": q_type,
                    "text": q_text,
                    "filename": f.name
                })
    
    # Sort by query number
    exam_items.sort(key=lambda x: x["number"])
    
    # Generate HTML content
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bộ Đề Thi Chính Thức - EBT Vision Search</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0f111a;
            color: #e2e8f0;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        header {
            margin-bottom: 40px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 {
            margin: 0;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
        }
        .back-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: #94a3b8;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .back-btn:hover {
            color: #fff;
            background: rgba(255,255,255,0.1);
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        .card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: flex-start;
            gap: 20px;
            backdrop-filter: blur(10px);
            transition: transform 0.2s, border-color 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.3);
        }
        .q-badge {
            font-family: 'JetBrains Mono', monospace;
            background: #1e293b;
            color: #3b82f6;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1rem;
            min-width: 60px;
            text-align: center;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        .type-badge {
            font-family: 'JetBrains Mono', monospace;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .type-KIS {
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        .type-QA {
            background: rgba(245, 158, 11, 0.1);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.2);
        }
        .type-TRAKE {
            background: rgba(139, 92, 246, 0.1);
            color: #8b5cf6;
            border: 1px solid rgba(139, 92, 246, 0.2);
        }
        .content {
            flex-grow: 1;
        }
        .text {
            font-size: 1.05rem;
            margin: 10px 0;
            color: #f1f5f9;
        }
        .meta-info {
            font-size: 0.85rem;
            color: #64748b;
        }
        .action-area {
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-self: center;
        }
        .btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: background 0.2s;
            white-space: nowrap;
        }
        .btn:hover {
            background: #1d4ed8;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Bộ Đề Thi Chính Thức (Sơ Tuyển 1)</h1>
                <p style="color: #94a3b8; margin: 5px 0 0 0;">Danh sách 25 câu hỏi đã được trích xuất từ SOTUYEN1-bo-de-thi.zip</p>
            </div>
            <a href="index.html" class="back-btn">← Trang tìm kiếm</a>
        </header>

        <div class="grid">
"""
    
    for item in exam_items:
        q_url = f"index.html?query={urllib.parse.quote(item['text'])}&type={item['type']}"
        html_content += f"""
            <div class="card">
                <div class="q-badge">#{item['number']}</div>
                <div class="content">
                    <span class="type-badge type-{item['type']}">{item['type']}</span>
                    <div class="text">{item['text']}</div>
                    <div class="meta-info">File: {item['filename']}</div>
                </div>
                <div class="action-area">
                    <a href="{q_url}" target="_blank" class="btn">🔍 Tìm Kiếm</a>
                </div>
            </div>
"""
            
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_html, "w", encoding="utf-8") as out:
        out.write(html_content)
        
    print(f"Successfully generated {output_html}")

if __name__ == "__main__":
    main()
