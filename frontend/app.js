document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const progressBar = document.getElementById('progressBar');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const fileList = document.getElementById('fileList');
    const clearDbBtn = document.getElementById('clearDbBtn');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatLog = document.getElementById('chatLog');
    const knowledgeBaseStatus = document.getElementById('knowledgeBaseStatus');
    const llmBadge = document.getElementById('llmBadge');

    // Evaluation Dashboard DOM Elements
    const btnOpenLatestEval = document.getElementById('btnOpenLatestEval');
    const evalModal = document.getElementById('evalModal');
    const btnCloseEvalModal = document.getElementById('btnCloseEvalModal');
    const activeMetricCard = document.getElementById('activeMetricCard');
    const retrievedChunksList = document.getElementById('retrievedChunksList');
    const evalTabButtons = document.querySelectorAll('.eval-tab-btn');

    // API endpoints
    const API_BASE = '/api';

    // State
    let uploadedFiles = [];
    let evaluationsCache = {};
    let latestMessageId = null;
    let currentActiveMetric = 'context_precision';
    let currentEvaluation = null;

    // Load active documents and configurations on start
    fetchDocuments();
    fetchConfig();

    async function fetchConfig() {
        try {
            const response = await fetch(`${API_BASE}/config`);
            if (response.ok) {
                const data = await response.json();
                const providerText = data.provider === 'ollama' ? 'Ollama' : 'Gemini';
                llmBadge.textContent = `${providerText}: ${data.model}`;
                if (data.provider === 'ollama') {
                    llmBadge.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
                    llmBadge.style.color = '#3B82F6';
                    llmBadge.style.borderColor = 'rgba(59, 130, 246, 0.2)';
                }
            }
        } catch (err) {
            console.error('Error fetching configuration:', err);
        }
    }

    // Config panel DOM elements
    const uploadConfigPanel = document.getElementById('uploadConfigPanel');
    const btnAutoSetup = document.getElementById('btnAutoSetup');
    const btnManualSetup = document.getElementById('btnManualSetup');
    const manualConfigArea = document.getElementById('manualConfigArea');
    const splitterSelect = document.getElementById('splitterSelect');
    const chunkSizeInput = document.getElementById('chunkSizeInput');
    const chunkSizeValue = document.getElementById('chunkSizeValue');
    const chunkOverlapInput = document.getElementById('chunkOverlapInput');
    const chunkOverlapValue = document.getElementById('chunkOverlapValue');
    const configError = document.getElementById('configError');
    const btnProcessDoc = document.getElementById('btnProcessDoc');
    const btnCancelUpload = document.getElementById('btnCancelUpload');
    const dropzoneMainText = document.getElementById('dropzoneMainText');
    const dropzoneSubText = document.getElementById('dropzoneSubText');

    let selectedFile = null;
    let selectedMode = 'automatic'; // 'automatic' or 'manual'

    // 1. File Selection & Dropzone Interactions
    dropzone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!selectedFile) {
            dropzone.classList.add('dragover');
        }
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (selectedFile) return;
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            prepareFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            prepareFile(fileInput.files[0]);
        }
    });

    // Prepare selected file, show config panel, and lock dropzone
    function prepareFile(file) {
        selectedFile = file;
        dropzoneMainText.textContent = file.name;
        dropzoneSubText.textContent = `Size: ${(file.size / 1024).toFixed(1)} KB (Ready to process)`;
        dropzone.style.cursor = 'default';
        
        // Show configuration panel
        uploadConfigPanel.style.display = 'flex';
        uploadStatus.textContent = '';
        uploadStatus.className = 'status-msg';
    }

    // Toggle Modes (Automatic vs Manual Setup)
    btnAutoSetup.addEventListener('click', () => {
        selectedMode = 'automatic';
        btnAutoSetup.classList.add('active');
        btnManualSetup.classList.remove('active');
        manualConfigArea.style.display = 'none';
        configError.style.display = 'none';
    });

    btnManualSetup.addEventListener('click', () => {
        selectedMode = 'manual';
        btnManualSetup.classList.add('active');
        btnAutoSetup.classList.remove('active');
        manualConfigArea.style.display = 'flex';
        validateConfig();
    });

    // Sync Sliders with Value Display Badges
    chunkSizeInput.addEventListener('input', () => {
        chunkSizeValue.textContent = chunkSizeInput.value;
        validateConfig();
    });

    chunkOverlapInput.addEventListener('input', () => {
        chunkOverlapValue.textContent = chunkOverlapInput.value;
        validateConfig();
    });

    // Configuration Parameter Validation
    function validateConfig() {
        if (selectedMode !== 'manual') return true;

        const size = parseInt(chunkSizeInput.value, 10);
        const overlap = parseInt(chunkOverlapInput.value, 10);

        if (overlap >= size) {
            configError.textContent = 'Validation Error: Chunk Overlap must be strictly less than Chunk Size.';
            configError.style.display = 'block';
            btnProcessDoc.disabled = true;
            btnProcessDoc.style.opacity = '0.5';
            btnProcessDoc.style.cursor = 'not-allowed';
            return false;
        } else {
            configError.style.display = 'none';
            btnProcessDoc.disabled = false;
            btnProcessDoc.style.opacity = '1';
            btnProcessDoc.style.cursor = 'pointer';
            return true;
        }
    }

    // Cancel Processing
    btnCancelUpload.addEventListener('click', resetUploadState);

    function resetUploadState() {
        selectedFile = null;
        fileInput.value = '';
        dropzoneMainText.textContent = 'Drag & drop files here';
        dropzoneSubText.textContent = 'or click to browse';
        dropzone.style.cursor = 'pointer';
        uploadConfigPanel.style.display = 'none';
        configError.style.display = 'none';
        
        // Reset defaults
        selectedMode = 'automatic';
        btnAutoSetup.classList.add('active');
        btnManualSetup.classList.remove('active');
        manualConfigArea.style.display = 'none';
        chunkSizeInput.value = 500;
        chunkSizeValue.textContent = 500;
        chunkOverlapInput.value = 50;
        chunkOverlapValue.textContent = 50;
        btnProcessDoc.disabled = false;
        btnProcessDoc.style.opacity = '1';
        btnProcessDoc.style.cursor = 'pointer';
    }

    // Process & Ingest File Trigger
    btnProcessDoc.addEventListener('click', async () => {
        if (!selectedFile) return;
        if (!validateConfig()) return;

        uploadStatus.className = 'status-msg loading';
        uploadStatus.textContent = `Processing "${selectedFile.name}"...`;
        progressBarContainer.style.display = 'block';
        progressBar.style.width = '20%';

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('setup_mode', selectedMode);

        if (selectedMode === 'manual') {
            formData.append('splitter_type', splitterSelect.value);
            formData.append('chunk_size', chunkSizeInput.value);
            formData.append('chunk_overlap', chunkOverlapInput.value);
        } else {
            // Defaults as specified in requirements
            formData.append('splitter_type', 'RecursiveCharacterTextSplitter');
            formData.append('chunk_size', '500');
            formData.append('chunk_overlap', '50');
        }

        try {
            progressBar.style.width = '50%';
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            progressBar.style.width = '90%';

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Processing failed');
            }

            progressBar.style.width = '100%';
            uploadStatus.className = 'status-msg success';
            uploadStatus.textContent = `Successfully processed and indexed ${selectedFile.name}!`;
            
            // Success reset
            setTimeout(() => {
                resetUploadState();
                fetchDocuments();
            }, 1500);

        } catch (err) {
            console.error('Processing error:', err);
            uploadStatus.className = 'status-msg error';
            uploadStatus.textContent = `Error: ${err.message}`;
        } finally {
            setTimeout(() => {
                progressBarContainer.style.display = 'none';
                progressBar.style.width = '0%';
            }, 1500);
        }
    });

    // 2. Fetch and List Documents
    async function fetchDocuments() {
        try {
            const response = await fetch(`${API_BASE}/documents`);
            if (response.ok) {
                const data = await response.json();
                uploadedFiles = data.documents || [];
                renderFileList();
                updateKBStatus();
            }
        } catch (err) {
            console.error('Error fetching documents:', err);
        }
    }

    function renderFileList() {
        fileList.innerHTML = '';
        if (uploadedFiles.length === 0) {
            fileList.innerHTML = '<li class="empty-list-msg">No files uploaded yet.</li>';
            return;
        }

        uploadedFiles.forEach(filename => {
            const li = document.createElement('li');
            li.className = 'file-item';
            
            // Icon
            const icon = document.createElement('span');
            icon.className = 'file-item-icon';
            icon.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10 9 9 9 8 9"/>
                </svg>
            `;
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'file-item-name';
            nameSpan.textContent = filename;
            nameSpan.title = filename;

            li.appendChild(icon);
            li.appendChild(nameSpan);
            fileList.appendChild(li);
        });
    }

    function updateKBStatus() {
        if (uploadedFiles.length === 0) {
            knowledgeBaseStatus.textContent = 'Knowledge base is empty';
        } else {
            knowledgeBaseStatus.textContent = `${uploadedFiles.length} file(s) available for context`;
        }
    }

    // 3. Clear Database
    clearDbBtn.addEventListener('click', async () => {
        if (uploadedFiles.length === 0) return;
        if (!confirm('Are you sure you want to clear the entire knowledge base? This cannot be undone.')) return;

        try {
            const response = await fetch(`${API_BASE}/clear`, { method: 'DELETE' });
            if (response.ok) {
                uploadedFiles = [];
                renderFileList();
                updateKBStatus();
                appendSystemMessage('Knowledge base has been cleared successfully.');
            }
        } catch (err) {
            console.error('Error clearing database:', err);
            alert('Failed to clear vector database.');
        }
    });

    // 4. Chat Log & Interaction
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Clear input
        userInput.value = '';

        // Remove welcome screen if it's there
        const welcome = document.querySelector('.welcome-container');
        if (welcome) welcome.remove();

        // Render User Message
        appendMessage(text, 'user');

        // Create Typing Indicator
        const typingIndicator = showTypingIndicator();

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            // Remove typing indicator
            typingIndicator.remove();

            if (!response.ok) {
                const errData = await response.json();
                appendMessage(`Error: ${errData.detail || 'Failed to generate response'}`, 'assistant');
                return;
            }

            const data = await response.json();
            
            // Cache evaluation results if present
            if (data.evaluation && data.message_id) {
                evaluationsCache[data.message_id] = data.evaluation;
                latestMessageId = data.message_id;
                btnOpenLatestEval.style.display = 'flex';
            }

            appendMessage(data.answer, 'assistant', data.sources, data.message_id);

        } catch (err) {
            console.error('Error sending message:', err);
            typingIndicator.remove();
            appendMessage('An error occurred while trying to connect to the backend server.', 'assistant');
        }
    });

    function appendMessage(text, sender, sources = [], messageId = null) {
        const row = document.createElement('div');
        row.className = `message-row ${sender}`;

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        // Simple markdown parsing helper for code block and strong tags
        const formattedText = parseMarkdown(text);
        bubble.innerHTML = `<div class="message-content">${formattedText}</div>`;

        // If assistant has sources, render collapsible list
        if (sender === 'assistant' && sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources-container';
            
            const uniqueId = 'sources_' + Math.random().toString(36).substr(2, 9);
            
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'sources-toggle';
            toggleBtn.innerHTML = `
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"/>
                </svg> View Sources (${sources.length})
            `;
            
            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';
            sourcesList.id = uniqueId;

            sources.forEach(src => {
                const item = document.createElement('div');
                item.className = 'source-item';
                
                const title = document.createElement('div');
                title.className = 'source-title';
                title.textContent = src.page ? `${src.source} (Page ${src.page})` : src.source;
                
                const snippet = document.createElement('div');
                snippet.className = 'source-snippet';
                snippet.textContent = `"${src.snippet}"`;

                item.appendChild(title);
                item.appendChild(snippet);
                sourcesList.appendChild(item);
            });

            toggleBtn.addEventListener('click', () => {
                sourcesList.classList.toggle('open');
                const isOpened = sourcesList.classList.contains('open');
                toggleBtn.querySelector('svg').style.transform = isOpened ? 'rotate(180deg)' : 'rotate(0)';
            });

            sourcesDiv.appendChild(toggleBtn);
            sourcesDiv.appendChild(sourcesList);
            bubble.appendChild(sourcesDiv);
        }

        // If assistant response contains evaluation, add evaluation details button inside message bubble
        if (sender === 'assistant' && messageId) {
            const evalBtn = document.createElement('button');
            evalBtn.className = 'btn-msg-eval';
            evalBtn.innerHTML = `
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg> View Evaluation
            `;
            evalBtn.addEventListener('click', () => {
                openEvaluationDashboard(messageId);
            });
            bubble.appendChild(evalBtn);
        }

        row.appendChild(bubble);
        chatLog.appendChild(row);
        scrollToBottom();
    }

    function appendSystemMessage(text) {
        const row = document.createElement('div');
        row.className = 'message-row assistant';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.style.borderStyle = 'dashed';
        bubble.style.borderColor = 'var(--text-dark)';
        bubble.innerHTML = `<div class="message-content" style="color: var(--text-muted); font-style: italic;">System: ${text}</div>`;
        row.appendChild(bubble);
        chatLog.appendChild(row);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'message-row assistant';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;

        bubble.appendChild(indicator);
        row.appendChild(bubble);
        chatLog.appendChild(row);
        scrollToBottom();

        return row;
    }

    function scrollToBottom() {
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Markdown Parser Helper
    function parseMarkdown(text) {
        let esc = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Block code: ```code```
        esc = esc.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        // Inline code: `code`
        esc = esc.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Bold: **text**
        esc = esc.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        // Bullet list
        esc = esc.replace(/^\s*[-*]\s+(.*)$/gm, '<li>$1</li>');
        esc = esc.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
        // Clean double nested lists
        esc = esc.replace(/<\/ul>\s*<ul>/g, '');

        // Newlines
        return esc.replace(/\n/g, '<br>');
    }

    // ============================================================================
    // 5. RAG Evaluation Dashboard Controls & Data Mapping
    // ============================================================================
    const METRICS_METADATA = {
        context_precision: {
            title: "Context Precision",
            category: "Retrieval Quality",
            description: "Measures whether the most relevant chunks are ranked at the top of the retrieved context list.",
            formula: "Context Precision @ K = ∑ (P@i * Relevance(i)) / Total Relevant retrieved chunks",
            evaluation: "Evaluated by computing the semantic cosine similarity of each retrieved chunk against the user query. The metric checks if chunks with higher similarity are positioned at earlier ranks.",
            inputs: "Query embedding and individual retrieved chunk embeddings.",
            meaning_high: "Excellent search ranking. The LLM gets the most relevant facts first, reducing prompt noise and generation latency.",
            meaning_low: "Poor ranking. Irrelevant chunks are placed before useful ones, which could mislead the LLM.",
            improvement: "Fine-tune the embeddings model, use a Cross-Encoder Re-ranker to reorder the chunks, or tune the similarity distance metric (e.g., switching from L2 to Cosine)."
        },
        context_recall: {
            title: "Context Recall",
            category: "Retrieval Quality",
            description: "Estimates the proportion of the necessary ground truth/relevant information that was successfully retrieved by the search pipeline.",
            formula: "Context Recall = |Relevant Retrieved Chunks| / |Total Estimated Relevant Chunks|",
            evaluation: "Calculated by identifying which retrieved chunks exceed a relevance similarity threshold (0.45) compared to the overall estimated relevance set.",
            inputs: "Query embedding and retrieved chunk embeddings.",
            meaning_high: "The retrieval system successfully gathered all key details required to fully address the user query.",
            meaning_low: "Crucial context was missed during vector lookup, likely causing the LLM to state that the answer is not in the context.",
            improvement: "Increase 'K' (number of retrieved chunks), adjust the chunk size to capture wider contexts, or use hybrid search (Vector + BM25 keyword matching)."
        },
        mrr: {
            title: "Mean Reciprocal Rank (MRR)",
            category: "Retrieval Quality",
            description: "Evaluates the rank order position of the first relevant chunk in the retrieved documents.",
            formula: "MRR = 1 / Rank of first relevant document",
            evaluation: "Determines the position of the first chunk that meets the query relevance threshold. If the first chunk is relevant, MRR is 1.0; if the second, it is 0.5; and so on.",
            inputs: "Rank order and similarity scores of retrieved chunks.",
            meaning_high: "The absolute best document matches are found immediately at index 0.",
            meaning_low: "The user has to scan far down the retrieved list to find useful facts, indicating weak top-1 retrieval performance.",
            improvement: "Implement semantic query expansion or query rewriting to align the question closer to the embedded document headers."
        },
        ndcg: {
            title: "Normalized Discounted Cumulative Gain (NDCG)",
            category: "Retrieval Quality",
            description: "Measures the overall gain of a document based on its relevance position, accounting for logarithmic ranking discounts.",
            formula: "NDCG = DCG / Ideal DCG",
            evaluation: "Compares the actual discounted cumulative gain of the retrieved rank positions against the mathematically ideal relevance sorting.",
            inputs: "Relevance relevance scores per rank.",
            meaning_high: "Highly relevant items are placed at the absolute top of the results list in a near-ideal order.",
            meaning_low: "Relevant items are pushed to lower ranks, suffering heavy position discounts.",
            improvement: "Apply query reranking models to maximize top-heavy sorting accuracy."
        },
        precision_at_k: {
            title: "Precision @ K",
            category: "Retrieval Quality",
            description: "Measures the fraction of retrieved chunks that are actually relevant to the user query.",
            formula: "Precision@K = Relevant Retrieved Chunks / K",
            evaluation: "Checks how many of the 'K' retrieved chunks meet the semantic relevance threshold.",
            inputs: "Retrieved documents list and relevance flags.",
            meaning_high: "Very low noise. Most of the retrieved documents inside the LLM prompt are directly related to the question.",
            meaning_low: "High clutter. The LLM prompt is filled with irrelevant chunks, increasing token cost and chance of distraction.",
            improvement: "Decrease K, use a higher similarity threshold filter, or refine document chunking strategies to avoid overlaps."
        },
        answer_relevancy: {
            title: "Answer Relevancy",
            category: "Generation Quality",
            description: "Assesses how relevant the generated answer is to the user's initial question, ignoring accuracy.",
            formula: "Answer Relevancy = Cosine_Similarity(Query_Embedding, Answer_Embedding)",
            evaluation: "Calculated using the local embeddings model to measure semantic alignment between the question vector and the generated answer vector.",
            inputs: "User query text and assistant generated response text.",
            meaning_high: "The assistant directly and specifically answered the question asked, without introducing unrelated topics.",
            meaning_low: "The assistant gave a generic response, avoided the question, or went off-topic.",
            improvement: "Refine system instructions to enforce direct, concise answers. Avoid conversational filler in LLM prompt templates."
        },
        faithfulness: {
            title: "Faithfulness",
            category: "Generation Quality",
            description: "Evaluates if the generated answer is grounded strictly in the retrieved context, indicating factual accuracy.",
            formula: "Faithfulness = Count(Claims supported by context) / Total Claims in Answer",
            evaluation: "Splits the response into sentences and computes the maximum semantic similarity of each sentence against the retrieved chunks. A low similarity identifies claims not present in the documents.",
            inputs: "Generated response sentences and retrieved chunk texts.",
            meaning_high: "Excellent grounding. All facts generated by the model can be traced back to the retrieved documents.",
            meaning_low: "High risk of hallucination. The model generated statements, figures, or claims not found in the documents.",
            improvement: "Set LLM temperature to 0.0, add strict system prompt constraints (e.g. 'Only use the context. Do not invent facts.'), or retrieve more chunks to fill gaps."
        },
        hallucination_score: {
            title: "Hallucination Detection",
            category: "Generation Quality",
            description: "Measures the degree to which the assistant's response relies on ungrounded external assertions or pre-training facts.",
            formula: "Hallucination Score = 1.0 - Faithfulness",
            evaluation: "Inverse of the Faithfulness score. Calculated from the mean of unsupported sentence percentages.",
            inputs: "Answer sentence alignments and retrieved chunks.",
            meaning_high: "The answer contains unverified facts or assumptions not present in the files.",
            meaning_low: "Zero hallucination detected. The response is perfectly grounded in your knowledge base.",
            improvement: "Enforce citation rules in prompt engineering and penalize the model for generating facts without document evidence."
        },
        context_utilization: {
            title: "Context Utilization",
            category: "Generation Quality",
            description: "Measures the efficiency of context usage by checking what percentage of retrieved chunks were actually used to synthesize the response.",
            formula: "Context Utilization = Chunks Used by LLM / Total Retrieved Chunks",
            evaluation: "Heuristically determined by checking if any sentence in the response has a strong semantic alignment (> 0.58) with the content of each chunk.",
            inputs: "Retrieved chunk metadata and generated response sentences.",
            meaning_high: "Highly efficient prompt. The LLM utilized most of the context provided to construct its response.",
            meaning_low: "Wasteful retrieval. Chunks were retrieved and sent to the LLM but never referenced, indicating unnecessary token overhead.",
            improvement: "Reduce retrieve count (K), improve query precision, or use summaries instead of raw chunks."
        },
        semantic_similarity: {
            title: "Semantic Similarity",
            category: "Generation Quality",
            description: "Measures the closeness of the semantic vector representation between the query and response.",
            formula: "Semantic Similarity = Cosine_Similarity(Query, Response)",
            evaluation: "Determines how closely aligned the meaning of the answer is with the user's intent.",
            inputs: "Query text and response text.",
            meaning_high: "The answer matches the semantic intent of the query.",
            meaning_low: "Mismatch in intent; the response might be a refusal or misunderstanding.",
            improvement: "Refine system context assembly to direct the LLM's attention to the user's specific query angles."
        },
        total_response_time: {
            title: "Total Response Time",
            category: "Performance & Resources",
            description: "The complete latency duration required to receive, process, retrieve, generate, and return the RAG answer.",
            formula: "Total Response Time = Retrieval Latency + LLM Response Time",
            evaluation: "Calculated using backend timestamps surrounding the entire chat process pipeline.",
            inputs: "System clock durations.",
            meaning_high: "Fast response. Good user experience.",
            meaning_low: "Slow response. The user suffers high wait times, often due to massive chunk counts or local model speed limits.",
            improvement: "Switch to a lighter local LLM (e.g. from 7B to 3B parameters), run Ollama on GPU/CUDA, or reduce vector search K."
        },
        retrieval_latency: {
            title: "Retrieval Latency",
            category: "Performance & Resources",
            description: "The time spent querying the Chroma vector database to find the K most relevant chunks.",
            formula: "Retrieval Latency = End Time of Db Query - Start Time of Db Query",
            evaluation: "Measured via Python system time directly surrounding the Chroma similarity search API call.",
            inputs: "Database query time metrics.",
            meaning_high: "Vector DB query is slow, possibly due to a massive collection, index overhead, or CPU constraints.",
            meaning_low: "Ultra-fast vector search, typical of lightweight Chroma DB instances.",
            improvement: "Ensure the Chroma DB directory is on an SSD, reduce the embedding vector size, or partition collections."
        },
        llm_response_time: {
            title: "LLM Response Time",
            category: "Performance & Resources",
            description: "The duration required by the local Ollama LLM to process prompt tokens and stream the completed answer.",
            formula: "LLM Response Time = End Time of LLM Call - Start Time of LLM Call",
            evaluation: "Measured via system time surrounding the LangChain chain execution.",
            inputs: "LLM generation time metrics.",
            meaning_high: "The local model is slow, usually due to running on CPU instead of GPU, or due to generating a very long text.",
            meaning_low: "Fast inference speed, indicating GPU hardware acceleration or concise model responses.",
            improvement: "Enable GPU offloading in Ollama, allocate more CPU cores, or reduce the max token output length limit."
        },
        total_tokens: {
            title: "Token Usage",
            category: "Performance & Resources",
            description: "The total token volume processed during the transaction, split into input context and generated output.",
            formula: "Total Tokens = Prompt (Input) Tokens + Completion (Output) Tokens",
            evaluation: "Computed locally using the tiktoken library tokenizer encoding.",
            inputs: "Prompt texts and generated completion string.",
            meaning_high: "Large prompts (many retrieved chunks) or long replies. High memory usage.",
            meaning_low: "Lighter prompts, matching optimal, concise text chunking.",
            improvement: "Reduce chunk size, decrease the context retrieve limit (K), or restrict LLM generation limits."
        },
        estimated_cost: {
            title: "Cost Estimation",
            category: "Performance & Resources",
            description: "The monetary cost associated with this transaction.",
            formula: "Cost = (Prompt Tokens * Input Rate) + (Completion Tokens * Output Rate)",
            evaluation: "Calculated based on standard model pricing. Since you are running a completely local Ollama instance, the cost is always $0.00.",
            inputs: "Token counts and provider pricing rates.",
            meaning_high: "A high volume of tokens was processed, which would be expensive on paid cloud APIs.",
            meaning_low: "Efficient transaction cost.",
            improvement: "Running Ollama locally is completely free ($0). If migrating to OpenAI, optimize token usage to lower costs."
        }
    };

    // Open Evaluation Dashboard for a specific message
    function openEvaluationDashboard(messageId) {
        currentEvaluation = evaluationsCache[messageId];
        if (!currentEvaluation) {
            console.error('No evaluation cached for message ID:', messageId);
            return;
        }

        evalModal.style.display = 'flex';
        renderActiveMetric();
        renderRetrievedChunks();
    }

    // Render active metric details
    function renderActiveMetric() {
        if (!currentEvaluation) return;

        const meta = METRICS_METADATA[currentActiveMetric];
        const score = currentEvaluation.metrics[currentActiveMetric];
        
        let displayScore = score;
        let scoreLabel = "";
        let colorClass = "";
        
        // Format based on metric type
        if (currentActiveMetric.includes('time')) {
            displayScore = `${score.toFixed(3)}s`;
            scoreLabel = "Latency";
            colorClass = score < 1.0 ? "badge-excellent" : (score < 3.0 ? "badge-good" : "badge-average");
        } else if (currentActiveMetric.includes('tokens')) {
            displayScore = score.toLocaleString();
            scoreLabel = "Tokens";
            colorClass = "badge-good";
        } else if (currentActiveMetric === 'estimated_cost') {
            displayScore = `$0.00`;
            scoreLabel = "Local Ollama";
            colorClass = "badge-excellent";
        } else {
            // Decimal score metrics [0, 1]
            const pct = (score * 100).toFixed(1);
            displayScore = `${pct}%`;
            
            if (score >= 0.85) {
                scoreLabel = "Excellent";
                colorClass = "badge-excellent";
            } else if (score >= 0.70) {
                scoreLabel = "Good";
                colorClass = "badge-good";
            } else if (score >= 0.50) {
                scoreLabel = "Average";
                colorClass = "badge-average";
            } else {
                scoreLabel = "Poor";
                colorClass = "badge-poor";
            }
        }

        const isTimeOrToken = currentActiveMetric.includes('time') || currentActiveMetric.includes('token') || currentActiveMetric === 'estimated_cost';
        const progressWidth = isTimeOrToken ? 100 : (score * 100);
        const barColorClass = isTimeOrToken ? "bg-good" : (score >= 0.85 ? "bg-excellent" : (score >= 0.70 ? "bg-good" : (score >= 0.50 ? "bg-average" : "bg-poor")));

        activeMetricCard.innerHTML = `
            <div class="metric-header">
                <div class="metric-title">
                    <span class="metric-category">${meta.category}</span>
                    <h3>${meta.title}</h3>
                </div>
                <div class="score-display-box">
                    <div class="score-large ${colorClass}">${displayScore}</div>
                    <div class="score-label ${colorClass}">${scoreLabel}</div>
                </div>
            </div>

            <div class="metric-progress-wrapper">
                <div class="metric-progress-bg">
                    <div class="metric-progress-fill ${barColorClass}" style="width: ${progressWidth}%"></div>
                </div>
            </div>

            <div class="metric-info-grid">
                <div class="info-block">
                    <h5>What is this metric?</h5>
                    <p>${meta.description}</p>
                </div>
                <div class="info-block">
                    <h5>How is it evaluated?</h5>
                    <p>${meta.evaluation}</p>
                </div>
                <div class="info-block">
                    <h5>Input Data Used</h5>
                    <p>${meta.inputs}</p>
                </div>
                <div class="info-block">
                    <h5>What does this score mean?</h5>
                    <p>${score >= 0.70 || isTimeOrToken ? meta.meaning_high : meta.meaning_low}</p>
                </div>
                
                <div class="formula-block">
                    <strong>Formula/Algorithm:</strong> ${meta.formula}
                </div>

                <div class="info-block" style="grid-column: span 2; border-top: 1px solid var(--border-color); padding-top: 14px; margin-top: 6px;">
                    <h5>How to improve this metric?</h5>
                    <p>${meta.improvement}</p>
                </div>
            </div>
        `;
    }

    // Render retrieved chunks checklist
    function renderRetrievedChunks() {
        if (!currentEvaluation || !currentEvaluation.chunks) return;

        retrievedChunksList.innerHTML = '';

        currentEvaluation.chunks.forEach(chunk => {
            const card = document.createElement('div');
            card.className = `chunk-card ${chunk.used_by_llm ? 'used' : 'ignored'}`;

            const badgeText = chunk.used_by_llm ? 'LLM Contributor' : 'Ignored by LLM';
            const badgeClass = chunk.used_by_llm ? 'used-badge' : 'ignored-badge';

            card.innerHTML = `
                <div class="chunk-card-header">
                    <div class="chunk-meta">
                        <strong>Chunk ${chunk.chunk_id}</strong>
                        <span>|</span>
                        <span>File: ${chunk.source_document}</span>
                        <span>|</span>
                        <span>Page: ${chunk.page}</span>
                    </div>
                    <span class="chunk-badge ${badgeClass}">${badgeText}</span>
                </div>
                <div class="chunk-text-box">
                    ${chunk.text}
                </div>
                <div class="chunk-explanation">
                    <strong>System Log:</strong> ${chunk.explanation}
                </div>
            `;
            retrievedChunksList.appendChild(card);
        });
    }

    // Sidebar tab buttons click listener
    evalTabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            evalTabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentActiveMetric = btn.getAttribute('data-metric');
            renderActiveMetric();
        });
    });

    // Close Modal Controls
    btnCloseEvalModal.addEventListener('click', () => {
        evalModal.style.display = 'none';
    });

    evalModal.addEventListener('click', (e) => {
        if (e.target === evalModal) {
            evalModal.style.display = 'none';
        }
    });

    // Open dashboard for latest response
    btnOpenLatestEval.addEventListener('click', () => {
        if (latestMessageId) {
            openEvaluationDashboard(latestMessageId);
        }
    });
});

