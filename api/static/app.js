/**
 * Main Application Logic for Video Retrieval AI System
 * EBT Project 2026
 */

const DEMO_MODE = false; // Set to true to show demo data if API fails

let searchInput, searchBtn, queryTypeBtns, loadingState, errorState, errorMessage, retryBtn, resultsContainer, resultsGrid, summaryBar, parsedInfoContainer;
let activeQueryType = 'KIS';
window.currentQueryData = null; // Store data for CSV export

function init() {
    searchInput = document.getElementById('search-input');
    searchBtn = document.getElementById('search-btn');
    queryTypeBtns = document.querySelectorAll('.query-type-btn');
    loadingState = document.getElementById('loading-state');
    errorState = document.getElementById('error-state');
    errorMessage = document.getElementById('error-message');
    retryBtn = document.getElementById('retry-btn');
    resultsContainer = document.getElementById('results-container');
    resultsGrid = document.getElementById('results-grid');
    summaryBar = document.getElementById('summary-bar');
    parsedInfoContainer = document.getElementById('parsed-info');

    setupEventListeners();
    checkSystemStatus();
    console.log('[EBT Search UI] Initialized successfully. Ready for queries.');
}

async function checkSystemStatus() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            const data = await res.json();
            if (statusText) statusText.textContent = data.searcher_loaded ? 'Sẵn sàng (LIVE)' : 'Sẵn sàng (FAISS)';
            if (statusDot) {
                statusDot.className = 'status-dot online';
                statusDot.style.background = '#22c55e';
            }
        }
    } catch (e) {
        if (statusText) statusText.textContent = 'Mất kết nối Backend';
        if (statusDot) {
            statusDot.className = 'status-dot offline';
            statusDot.style.background = '#ef4444';
        }
    }
}

window.switchTrack = function(trackType) {
    activeQueryType = trackType.toUpperCase();
    if (!queryTypeBtns) queryTypeBtns = document.querySelectorAll('.query-type-btn');
    if (!searchInput) searchInput = document.getElementById('search-input');
    
    queryTypeBtns.forEach(b => {
        const t = b.dataset.type || b.textContent.trim().toUpperCase();
        if (t === activeQueryType) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });

    if (searchInput) {
        if (activeQueryType === 'KIS') {
            searchInput.placeholder = 'Nhập truy vấn (Tiếng Việt / English)... VD: a man in a red shirt / người đàn ông áo đỏ';
        } else if (activeQueryType === 'QA') {
            searchInput.placeholder = 'Nhập câu hỏi (VIE / ENG)... VD: What is the man doing? / Người đó làm gì?';
        } else if (activeQueryType === 'TRAKE') {
            searchInput.placeholder = 'Nhập truy vấn tracking (VIE / ENG)... VD: Step 1: running, Step 2: jumping';
        }
    }
};

function setupEventListeners() {
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        searchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            performSearch();
        });
    } else {
        if (searchBtn) searchBtn.addEventListener('click', performSearch);
        if (searchInput) {
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    performSearch();
                }
            });
        }
    }

    if (queryTypeBtns) {
        queryTypeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.currentTarget;
                const type = target.dataset.type || target.textContent.trim().toUpperCase();
                window.switchTrack(type);
            });
        });
    }

    if (retryBtn) retryBtn.addEventListener('click', performSearch);

    // Image error delegation
    if (resultsGrid) {
        resultsGrid.addEventListener('error', (e) => {
            if (e.target.tagName && e.target.tagName.toLowerCase() === 'img') {
                handleImageError(e.target);
            }
        }, true);
    }
}
window.performSearch = performSearch;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

let searchProgressTimer = null;

/**
 * Handles search execution
 */
async function performSearch() {
    if (!searchInput) searchInput = document.getElementById('search-input');
    if (!searchBtn) searchBtn = document.getElementById('search-btn');
    if (!loadingState) loadingState = document.getElementById('loading-state');
    const loadingStatusText = document.getElementById('loading-status-text');
    if (!errorState) errorState = document.getElementById('error-state');
    if (!resultsContainer) resultsContainer = document.getElementById('results-container');
    if (!resultsGrid) resultsGrid = document.getElementById('results-grid');
    if (!searchInput) return;
    
    const query = searchInput.value.trim();
    console.log('[EBT Search UI] Triggered search:', { query, activeQueryType });
    if (!query) {
        showError('Vui lòng nhập nội dung tìm kiếm');
        return;
    }

    // UI updates for loading
    if (searchBtn) {
        searchBtn.disabled = true;
        searchBtn.classList.add('loading');
    }
    if (loadingState) loadingState.style.display = 'flex';
    if (errorState) errorState.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (resultsGrid) resultsGrid.innerHTML = '';

    // Stage progression animation
    const stages = [
        "Đang phân tích câu truy vấn (Gemini NLP Compiler)...",
        "Đang truy vết không gian đặc trưng SigLIP2 & FAISS...",
        "Đang đối soát nhận diện đối tượng & bộ lọc không gian...",
        "Đang tinh chỉnh xếp hạng & tổng hợp kết quả..."
    ];
    let stageIdx = 0;
    if (loadingStatusText) loadingStatusText.textContent = stages[0];
    clearInterval(searchProgressTimer);
    searchProgressTimer = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        if (loadingStatusText) loadingStatusText.textContent = stages[stageIdx];
    }, 2500);

    let endpoint = '/api/v1/search/kis';
    let requestBody = {
        query: query,
        query_type: activeQueryType,
        top_k: 100
    };

    if (activeQueryType === 'QA') {
        requestBody.question = query;
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errJson = await response.json().catch(() => null);
            const errDetail = errJson && errJson.detail ? errJson.detail : `${response.status} ${response.statusText}`;
            throw new Error(`Lỗi Server: ${errDetail}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (error) {
        console.error('Search error:', error);
        if (DEMO_MODE) {
            console.info('Falling back to demo mode data...');
            renderResults(getDemoData(query, activeQueryType));
        } else {
            showError('Không thể thực hiện truy vấn: ' + error.message);
        }
    } finally {
        clearInterval(searchProgressTimer);
        if (searchBtn) {
            searchBtn.disabled = false;
            searchBtn.classList.remove('loading');
        }
        if (loadingState) loadingState.style.display = 'none';
    }
}

    /**
     * Renders the search results
     * @param {Object} data - The API response data
     */
    function renderResults(data) {
        if (!data || !data.results || data.results.length === 0) {
            showError('Không tìm thấy kết quả phù hợp');
            return;
        }

        if (resultsContainer) resultsContainer.style.display = 'block';
        
        // Render summary
        if (summaryBar) {
            window.currentQueryData = data; // Save for export
            
            summaryBar.innerHTML = `
                <div class="summary-item">
                    <span class="summary-label">Kết quả:</span>
                    <span class="summary-value">${data.total_results || 0}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Thời gian:</span>
                    <span class="summary-value">${formatTime(data.search_time_ms || 0)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Cache:</span>
                    <span class="summary-value ${data.cache_hit ? 'cache-hit' : 'cache-miss'}">${data.cache_hit ? 'HIT' : 'MISS'}</span>
                </div>
                <div class="summary-item" style="margin-left: auto;">
                    <button class="secondary-btn" onclick="exportSubmissionCSV()" style="padding: 4px 12px; font-size: 0.85em; background: #0284c7; color: #fff; border:none; border-radius: 4px; cursor: pointer;">⬇️ Xuất CSV (AIC Format)</button>
                </div>
            `;
        }

        // Render parsed info
        if (parsedInfoContainer) {
            if (data.parsed_info) {
                let tagsHtml = '';
                if (data.parsed_info.extracted_objects && data.parsed_info.extracted_objects.length > 0) {
                    tagsHtml = data.parsed_info.extracted_objects.map(obj => `<span class="tag obj-tag">${escapeHtml(obj)}</span>`).join('');
                }
                parsedInfoContainer.innerHTML = `
                    <div class="parsed-item">
                        <strong>Nội dung:</strong> <span>${escapeHtml(data.parsed_info.normalized_text || data.query)}</span>
                    </div>
                    ${tagsHtml ? `<div class="parsed-item"><strong>Đối tượng:</strong> <div class="tags-container">${tagsHtml}</div></div>` : ''}
                `;
                parsedInfoContainer.style.display = 'flex';
            } else {
                parsedInfoContainer.style.display = 'none';
            }
        }

        // Render cards
        if (resultsGrid) {
            data.results.forEach((result, index) => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.style.animationDelay = `${index * 0.05}s`;
                
                let rankClass = 'rank-default';
                if (result.rank === 1) rankClass = 'rank-gold';
                else if (result.rank >= 2 && result.rank <= 5) rankClass = 'rank-silver';

                const labelsHtml = (result.detected_labels || []).map(label => 
                    `<span class="pill">${escapeHtml(label)}</span>`
                ).join('');

                const watchUrlHtml = result.watch_url ? 
                    `<a href="${escapeHtml(result.watch_url)}" target="_blank" rel="noopener noreferrer" class="tester-verify-btn" title="Mở YouTube tại đúng phút:giây">
                        🎬 Verify Video @ ⏱️ ${escapeHtml(result.timestamp || '00:00')} (${result.pts_time}s)
                     </a>` : 
                    `<span class="tester-verify-btn disabled">
                        ⏱️ ${escapeHtml(result.timestamp || '00:00')} (${result.pts_time}s)
                     </span>`;

                const videoTitleHtml = result.video_title && result.video_title !== result.video_id ?
                    `<div class="video-title" title="${escapeHtml(result.video_title)}">${escapeHtml(result.video_title)}</div>` : '';

                // Bug #1 Fix: Show VQA Answer ONLY for Q&A and TRAKE, NOT for KIS.
                // KIS = Known Item Search — VLM answer is a description, not a Q&A answer.
                // Displaying it for KIS is misleading (looks like a wrong answer field).
                const isQAmode = data.query_type && data.query_type !== 'KIS';
                const vqaAnswerHtml = (result.vqa_answer && isQAmode) ? 
                    `<div style="margin-top: 8px; padding: 6px; background-color: #ccfbf1; color: #0f766e; border-left: 3px solid #14b8a6; border-radius: 4px; font-weight: bold; font-size: 0.9em;">
                        Answer: ${escapeHtml(result.vqa_answer)}
                     </div>` : '';

                const trakeSegmentHtml = (data.query_type === 'TRAKE' && result.start_frame !== undefined) ?
                    `<div style="margin-top: 8px; padding: 6px; background-color: #fef08a; color: #854d0e; border-left: 3px solid #eab308; border-radius: 4px; font-weight: bold; font-size: 0.9em;">
                        TRAKE Segment: [${result.start_frame}, ${result.end_frame}]
                     </div>` : '';

                card.innerHTML = `
                    <div class="card-image-container">
                        <span class="rank-badge ${rankClass}">#${result.rank}</span>
                        <img src="${escapeHtml(result.frame_url)}" alt="Frame ${result.frame_idx}" class="frame-image" data-video="${escapeHtml(result.video_id)}" loading="lazy">
                    </div>
                    <div class="card-content">
                        <div class="card-header">
                            <h3 class="video-id">${escapeHtml(result.video_id)}</h3>
                            <span class="frame-idx">Frame #${result.frame_idx}</span>
                        </div>
                        ${videoTitleHtml}
                        ${vqaAnswerHtml}
                        ${trakeSegmentHtml}
                        <div class="tester-verify-container">
                            ${watchUrlHtml}
                        </div>
                        
                        <div class="feedback-actions" style="margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap;">
                            <button class="feedback-btn match-btn" onclick="submitFeedback('${data.query_id}', '${result.video_id}', ${result.frame_idx}, 'MATCH', ${result.siglip_score ?? 0}, ${result.obj_score ?? 0}, ${result.spatial_score ?? 0}, ${result.fusion_score ?? 0}, this, '${(data.query || '').replace(/'/g, "\\\\'")}')" style="background: #15803d; color: white; border: 1px solid #22c55e; padding: 4px 8px; border-radius: 4px; font-size: 0.78em; font-weight: 600; cursor: pointer;">✅ Match</button>
                            <button class="feedback-btn uncertain-btn" onclick="submitFeedback('${data.query_id}', '${result.video_id}', ${result.frame_idx}, 'UNCERTAIN', ${result.siglip_score ?? 0}, ${result.obj_score ?? 0}, ${result.spatial_score ?? 0}, ${result.fusion_score ?? 0}, this, '${(data.query || '').replace(/'/g, "\\\\'")}')" style="background: #b45309; color: white; border: 1px solid #f59e0b; padding: 4px 8px; border-radius: 4px; font-size: 0.78em; font-weight: 600; cursor: pointer;">⚠️ Không chắc</button>
                            <button class="feedback-btn mismatch-btn" onclick="submitFeedback('${data.query_id}', '${result.video_id}', ${result.frame_idx}, 'MISMATCH', ${result.siglip_score ?? 0}, ${result.obj_score ?? 0}, ${result.spatial_score ?? 0}, ${result.fusion_score ?? 0}, this, '${(data.query || '').replace(/'/g, "\\\\'")}')" style="background: #b91c1c; color: white; border: 1px solid #ef4444; padding: 4px 8px; border-radius: 4px; font-size: 0.78em; font-weight: 600; cursor: pointer;">❌ Mismatch</button>
                        </div>
                        
                        <div class="scores-container">
                            ${createScoreBar('Fusion', result.fusion_score, 1.0, 'fusion-bar')}
                            ${createScoreBar('SIGLIP', result.siglip_score ?? 0.0, 0.3, 'clip-bar')}
                            ${createScoreBar('Object', result.obj_score, 1.0, 'obj-bar')}
                            ${createScoreBar('Spatial', result.spatial_score || 0.0, 1.0, 'spatial-bar')}
                        </div>
                        
                        ${labelsHtml ? `<div class="labels-container">${labelsHtml}</div>` : ''}
                    </div>
                `;
                
                resultsGrid.appendChild(card);
            });
        }

        // Trigger animations for bars
        setTimeout(() => {
            const progressBars = document.querySelectorAll('.progress-fill');
            progressBars.forEach(bar => {
                bar.style.width = bar.dataset.width || '0%';
            });
        }, 100);
    }

    /**
     * Creates HTML for a score bar
     */
    function createScoreBar(label, value, maxValue, colorClass) {
        const val = value || 0;
        const percentage = Math.min(100, Math.max(0, (val / maxValue) * 100));
        return `
            <div class="score-row">
                <span class="score-label">${escapeHtml(label)}</span>
                <div class="progress-track">
                    <div class="progress-fill ${colorClass}" data-width="${percentage}%" style="width: 0%"></div>
                </div>
                <span class="score-value">${val.toFixed(4)}</span>
            </div>
        `;
    }

    /**
     * Displays an error message
     * @param {string} msg 
     */
    function showError(msg) {
        if (errorState) errorState.style.display = 'block';
        if (errorMessage) errorMessage.textContent = msg;
        if (resultsContainer) resultsContainer.style.display = 'none';
    }

    /**
     * Handles broken image links — removes img entirely to prevent any retry loop
     */
    function handleImageError(imgElement) {
        // Guard: only handle once per image
        if (imgElement.dataset.errorHandled) return;
        imgElement.dataset.errorHandled = 'true';

        const videoId = imgElement.dataset.video || 'N/A';
        
        const placeholder = document.createElement('div');
        placeholder.className = 'image-placeholder';
        placeholder.innerHTML = `
            <div class="placeholder-content">
                <span class="placeholder-icon">🎥</span>
                <span class="placeholder-text">${escapeHtml(videoId)}</span>
                <span class="placeholder-hint">Không tải được ảnh</span>
            </div>
        `;
        
        // Replace the img element entirely (not just hide)
        if (imgElement.parentNode) {
            imgElement.parentNode.replaceChild(placeholder, imgElement);
        }
    }

    // Utilities
    function formatTime(ms) {
        if (ms < 1000) return `${ms.toFixed(1)}ms`;
        return `${(ms / 1000).toFixed(2)}s`;
    }

    function escapeHtml(unsafe) {
        if (unsafe == null) return '';
        return unsafe
             .toString()
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    /**
     * Generates demo data for mock mode
     */
    function getDemoData(query, type) {
        const results = [];
        for (let i = 1; i <= 20; i++) {
            const clipScore = Math.random() * 0.4 + 0.1;
            const objScore = Math.random() * 0.5 + 0.1;
            const fusionScore = (clipScore * 0.6) + (objScore * 0.4);
            
            results.push({
                rank: i,
                video_id: `L25_V${String(Math.floor(Math.random() * 100)).padStart(3, '0')}`,
                frame_idx: Math.floor(Math.random() * 30000),
                clip_score: clipScore,
                obj_score: objScore,
                spatial_score: 0.0,
                fusion_score: fusionScore,
                frame_url: 'non_existent_image_to_trigger_fallback.jpg',
                detected_labels: ['person', 'clothing', 'car', 'tree'].sort(() => 0.5 - Math.random()).slice(0, Math.floor(Math.random() * 3) + 1)
            });
        }
        
        results.sort((a, b) => b.fusion_score - a.fusion_score);
        results.forEach((r, idx) => r.rank = idx + 1);

        return {
            query_id: `demo_${Math.random().toString(36).substr(2, 9)}`,
            query: query,
            query_type: type,
            total_results: 20,
            search_time_ms: Math.random() * 500 + 50,
            cache_hit: Math.random() > 0.7,
            parsed_info: {
                normalized_text: query.toLowerCase(),
                extracted_objects: ['person', 'clothing'],
                sub_events: []
            },
            results: results
        };
    }

/**
 * Global Handler for User Feedback / Ground Truth Annotation
 */
window.submitFeedback = async function(queryId, videoId, frameIdx, verdict, siglipScore, objScore, spatialScore, fusionScore, btnElement, queryText = "") {
    try {
        const payload = {
            query_id: queryId,
            query_text: queryText || "",
            video_id: videoId,
            frame_idx: frameIdx,
            verdict: verdict,
            siglip_score: siglipScore,
            obj_score: objScore,
            spatial_score: spatialScore,
            fusion_score: fusionScore
        };

        const res = await fetch('/api/v1/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        // Highlight card border
        const card = btnElement.closest('.result-card');
        if (card) {
            card.style.border = verdict === 'MATCH' ? '2px solid #22c55e' : (verdict === 'UNCERTAIN' ? '2px solid #f59e0b' : '2px solid #ef4444');
            card.style.backgroundColor = verdict === 'MATCH' ? 'rgba(34, 197, 94, 0.08)' : (verdict === 'UNCERTAIN' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(239, 68, 68, 0.08)');
        }

        // Visual feedback on button
        const container = btnElement.parentElement;
        container.querySelectorAll('button').forEach(b => b.style.opacity = '0.4');
        btnElement.style.opacity = '1.0';
        btnElement.style.transform = 'scale(1.05)';

    } catch (err) {
        console.error('Failed to submit feedback:', err);
        alert('Lỗi lưu kết quả chấm: ' + err.message);
    }
};


/**
 * Export current results to AIC format CSV
 */
window.exportSubmissionCSV = function() {
    const data = window.currentQueryData;
    if (!data || !data.results || data.results.length === 0) {
        alert("Không có dữ liệu để xuất CSV!");
        return;
    }
    
    let csvContent = "";
    if (data.query_type === 'QA') {
        // AIC QA Format: <video_id>,<frame_idx>,"<answer>" (No header)
        data.results.forEach(r => {
            const ans = (r.vqa_answer || "").replace(/"/g, '""');
            csvContent += `${r.video_id},${r.frame_idx},"${ans}"\r\n`;
        });
    } else if (data.query_type === 'TRAKE') {
        // AIC TRAKE Format: <video_id>,<e1>,<e2>,<e3>,<e4> (No header)
        data.results.forEach(r => {
            const f = r.frame_idx || 0;
            csvContent += `${r.video_id},${f},${f+25},${f+50},${f+75}\r\n`;
        });
    } else {
        // AIC KIS Format: <video_id>,<frame_idx> (No header)
        data.results.forEach(r => {
            csvContent += `${r.video_id},${r.frame_idx}\r\n`;
        });
    }
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `submission_${data.query_id || 'results'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};
