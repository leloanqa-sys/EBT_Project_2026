/**
 * Main Application Logic for Video Retrieval AI System
 * EBT Project 2026
 */

const DEMO_MODE = true; // Set to true to show demo data if API fails

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const queryTypeBtns = document.querySelectorAll('.query-type-btn');
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    const retryBtn = document.getElementById('retry-btn');
    const resultsContainer = document.getElementById('results-container');
    const resultsGrid = document.getElementById('results-grid');
    const summaryBar = document.getElementById('summary-bar');
    const parsedInfoContainer = document.getElementById('parsed-info');

    let activeQueryType = 'KIS';

    // Initialization
    init();

    function init() {
        setupEventListeners();
    }

    function setupEventListeners() {
        if (searchBtn) searchBtn.addEventListener('click', performSearch);
        
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
        }

        if (queryTypeBtns) {
            queryTypeBtns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    queryTypeBtns.forEach(b => b.classList.remove('active'));
                    const target = e.currentTarget;
                    target.classList.add('active');
                    activeQueryType = target.dataset.type || target.textContent.trim().toUpperCase();
                    
                    // Update placeholder based on type
                    if (searchInput) {
                        if (activeQueryType === 'KIS') {
                            searchInput.placeholder = 'Nhập truy vấn (Tiếng Việt / English)... VD: a man in a red shirt / người đàn ông áo đỏ';
                        } else if (activeQueryType === 'QA') {
                            searchInput.placeholder = 'Nhập câu hỏi (VIE / ENG)... VD: What is the man doing? / Người đó làm gì?';
                        } else if (activeQueryType === 'TRAKE') {
                            searchInput.placeholder = 'Nhập truy vấn tracking (VIE / ENG)... VD: Step 1: running, Step 2: jumping';
                        }
                    }
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

    /**
     * Handles search execution
     */
    async function performSearch() {
        if (!searchInput) return;
        
        const query = searchInput.value.trim();
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

        const requestBody = {
            query: query,
            query_type: activeQueryType,
            top_k: 100
        };

        try {
            const response = await fetch('/api/v1/search/kis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Lỗi Server: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error('Search error:', error);
            if (DEMO_MODE) {
                console.info('Falling back to demo mode data...');
                renderResults(getDemoData(query, activeQueryType));
            } else {
                showError('Không thể kết nối đến máy chủ. Vui lòng thử lại sau. ' + error.message);
            }
        } finally {
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
                        <div class="tester-verify-container">
                            ${watchUrlHtml}
                        </div>
                        
                        <div class="feedback-actions" style="margin-top: 8px; display: flex; gap: 8px;">
                            <button class="feedback-btn match-btn" onclick="submitFeedback('${data.query_id}', '${result.video_id}', ${result.frame_idx}, 1, ${result.clip_score}, ${result.obj_score}, ${result.spatial_score}, ${result.has_target_objects}, this)" style="background: #28a745; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer;">✅ Match</button>
                            <button class="feedback-btn mismatch-btn" onclick="submitFeedback('${data.query_id}', '${result.video_id}', ${result.frame_idx}, 0, ${result.clip_score}, ${result.obj_score}, ${result.spatial_score}, ${result.has_target_objects}, this)" style="background: #dc3545; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer;">❌ Mismatch</button>
                        </div>
                        
                        <div class="scores-container">
                            ${createScoreBar('Fusion', result.fusion_score, 1.0, 'fusion-bar')}
                            ${createScoreBar('CLIP', result.clip_score, 1.0, 'clip-bar')}
                            ${createScoreBar('Object', result.obj_score, 1.0, 'obj-bar')}
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
                normalized_text: query.toLowerCase().replace(/[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ ]/g, ''),
                extracted_objects: ['person', 'clothing'],
                sub_events: []
            },
            results: results
        };
    }
});
