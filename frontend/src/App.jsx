import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking' | 'online' | 'offline'
  const [toolsData, setToolsData] = useState({ total_tools: 0, servers: {}, tools: [] });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showToolsModal, setShowToolsModal] = useState(false);

  // RAG Document state
  const [threadDocuments, setThreadDocuments] = useState([]);
  const [isUploadingPdf, setIsUploadingPdf] = useState(false);

  // User API Keys state (persisted in browser localStorage)
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [openaiKeyInput, setOpenaiKeyInput] = useState(() => localStorage.getItem('chatbot_openai_key') || '');
  const [hfTokenInput, setHfTokenInput] = useState(() => localStorage.getItem('chatbot_hf_token') || '');
  const [savedOpenaiKey, setSavedOpenaiKey] = useState(() => localStorage.getItem('chatbot_openai_key') || '');
  const [savedHfToken, setSavedHfToken] = useState(() => localStorage.getItem('chatbot_hf_token') || '');
  const [showOpenaiPassword, setShowOpenaiPassword] = useState(false);
  const [showHfPassword, setShowHfPassword] = useState(false);
  const [settingsStatusMessage, setSettingsStatusMessage] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll to bottom on messages update
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Initial load
  useEffect(() => {
    checkHealth();
    fetchThreads();
    fetchTools();
  }, []);

  // Fetch thread documents whenever activeThreadId changes
  useEffect(() => {
    if (activeThreadId) {
      fetchThreadDocuments(activeThreadId);
    } else {
      setThreadDocuments([]);
    }
  }, [activeThreadId]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    } catch {
      setBackendStatus('offline');
    }
  };

  const fetchThreads = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/threads`);
      if (res.ok) {
        const data = await res.json();
        setThreads(data.threads || []);
      }
    } catch {
      // Backend may be starting
    }
  };

  const fetchTools = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/tools`);
      if (res.ok) {
        const data = await res.json();
        setToolsData(data);
      }
    } catch {
      // Handled silently
    }
  };

  const fetchThreadDocuments = async (threadId) => {
    try {
      const res = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}/documents`);
      if (res.ok) {
        const data = await res.json();
        setThreadDocuments(data.documents || []);
      }
    } catch {
      setThreadDocuments([]);
    }
  };

  const handleSelectThread = async (threadId) => {
    if (threadId === activeThreadId) return;

    setIsLoadingHistory(true);
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      const res = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}`);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Failed to load conversation (${res.status})`);
      }
      const data = await res.json();
      setActiveThreadId(data.thread_id);
      setMessages(data.messages || []);
      setIsSidebarOpen(false);
    } catch (err) {
      setErrorMessage(err.message || 'Error loading conversation history');
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleNewChat = () => {
    setActiveThreadId(null);
    setMessages([]);
    setErrorMessage(null);
    setSuccessMessage(null);
    setThreadDocuments([]);
    setIsSidebarOpen(false);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const handleSaveApiKeys = (e) => {
    if (e) e.preventDefault();
    const cleanOpenai = openaiKeyInput.trim();
    const cleanHf = hfTokenInput.trim();

    if (cleanOpenai) {
      localStorage.setItem('chatbot_openai_key', cleanOpenai);
    } else {
      localStorage.removeItem('chatbot_openai_key');
    }

    if (cleanHf) {
      localStorage.setItem('chatbot_hf_token', cleanHf);
    } else {
      localStorage.removeItem('chatbot_hf_token');
    }

    setSavedOpenaiKey(cleanOpenai);
    setSavedHfToken(cleanHf);
    setSettingsStatusMessage('API keys saved in browser storage!');
    setTimeout(() => setSettingsStatusMessage(null), 3000);
  };

  const handleClearApiKeys = () => {
    localStorage.removeItem('chatbot_openai_key');
    localStorage.removeItem('chatbot_hf_token');
    setOpenaiKeyInput('');
    setHfTokenInput('');
    setSavedOpenaiKey('');
    setSavedHfToken('');
    setSettingsStatusMessage('Cleared custom keys. Using server environment defaults.');
    setTimeout(() => setSettingsStatusMessage(null), 3000);
  };

  const hasCustomKeys = Boolean(savedOpenaiKey || savedHfToken);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage('Only PDF documents (.pdf) can be uploaded for RAG indexing.');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    // Ensure we have an active thread ID; generate one if not present
    let targetThreadId = activeThreadId;
    if (!targetThreadId) {
      targetThreadId = 'session-' + Math.random().toString(36).substring(2, 10);
      setActiveThreadId(targetThreadId);
    }

    setIsUploadingPdf(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    const formData = new FormData();
    formData.append('file', file);

    const headers = {};
    if (savedOpenaiKey) {
      headers['X-OpenAI-API-Key'] = savedOpenaiKey;
    }

    try {
      const res = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(targetThreadId)}/documents`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      setSuccessMessage(`Indexed "${data.filename}" (${data.pages} pages, ${data.chunks} chunks). Ready for questions!`);
      fetchThreadDocuments(targetThreadId);
      fetchThreads();
    } catch (err) {
      setErrorMessage(err.message || 'Failed to upload and index PDF.');
    } finally {
      setIsUploadingPdf(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDeleteDocument = async (docId, filename) => {
    if (!activeThreadId) return;
    const confirmed = window.confirm(`Remove "${filename}" from this chat thread?`);
    if (!confirmed) return;

    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const res = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(activeThreadId)}/documents/${encodeURIComponent(docId)}`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to delete document.');
      }

      setSuccessMessage(`Removed "${filename}".`);
      fetchThreadDocuments(activeThreadId);
    } catch (err) {
      setErrorMessage(err.message || 'Error removing document.');
    }
  };

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    const trimmed = inputValue.trim();
    if (!trimmed || isLoading) return;

    setErrorMessage(null);

    const optimisticUserMessage = { role: 'user', content: trimmed };
    setMessages((prev) => [...prev, optimisticUserMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const payload = {
        message: trimmed,
        ...(activeThreadId ? { thread_id: activeThreadId } : {}),
        ...(savedOpenaiKey ? { openai_api_key: savedOpenaiKey } : {}),
        ...(savedHfToken ? { hf_token: savedHfToken } : {}),
      };

      const headers = {
        'Content-Type': 'application/json',
      };
      if (savedOpenaiKey) headers['X-OpenAI-API-Key'] = savedOpenaiKey;
      if (savedHfToken) headers['X-HF-Token'] = savedHfToken;

      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Request failed with status ${res.status}`);
      }

      const data = await res.json();

      if (data.thread_id) {
        setActiveThreadId(data.thread_id);
      }

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ]);

      fetchThreads();
      fetchTools();
      setBackendStatus('online');
    } catch (err) {
      setErrorMessage(err.message || 'Failed to send message. Please ensure backend is running.');
      setBackendStatus('offline');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePromptClick = (prompt) => {
    setInputValue(prompt);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const mcpToolsCount = toolsData.mcp_tools_count || 0;
  const serversList = Object.values(toolsData.servers || {});

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-icon">⚡</div>
            <span className="brand-title">LangGraph AI + RAG</span>
          </div>
        </div>

        <button
          className="new-chat-btn"
          id="new-chat-button"
          onClick={handleNewChat}
        >
          <span>＋</span> New Conversation
        </button>

        {/* Persistent PDF RAG Widget */}
        <div className="sidebar-rag-card">
          <div className="sidebar-tools-header">
            <span className="sidebar-tools-title">
              Thread PDFs ({threadDocuments.length})
            </span>
            <button
              className="tools-badge-btn upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploadingPdf}
              title="Upload PDF document to index for this thread"
            >
              {isUploadingPdf ? 'Indexing…' : '＋ Add PDF'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
            />
          </div>

          <div className="rag-docs-list">
            {isUploadingPdf && (
              <div className="rag-uploading-indicator">
                <div className="spinner small"></div>
                <span>Extracting & indexing document...</span>
              </div>
            )}
            {threadDocuments.length === 0 && !isUploadingPdf ? (
              <div className="empty-rag-text">No PDFs uploaded for this thread</div>
            ) : (
              threadDocuments.map((doc) => (
                <div key={doc.document_id} className="rag-doc-item">
                  <div className="rag-doc-info" title={doc.filename}>
                    <span className="rag-doc-icon">📄</span>
                    <span className="rag-doc-name">{doc.filename}</span>
                    <span className="rag-doc-meta">{doc.pages}p • {doc.chunks}c</span>
                  </div>
                  <button
                    className="rag-doc-delete"
                    onClick={() => handleDeleteDocument(doc.document_id, doc.filename)}
                    title="Delete document"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Tools & MCP Ecosystem Card */}
        <div className="sidebar-tools-card">
          <div className="sidebar-tools-header">
            <span className="sidebar-tools-title">Active Tools ({toolsData.total_tools || 4})</span>
            <button
              className="tools-badge-btn"
              onClick={() => setShowToolsModal(true)}
              title="View all registered tools and servers"
            >
              Details
            </button>
          </div>

          <div className="server-chips-list">
            <div className="server-chip">
              <span className="dot online"></span>
              <span>Native Tools (4 incl. RAG)</span>
            </div>
            {serversList.map((srv) => (
              <div
                key={srv.name}
                className={`server-chip ${srv.connected ? 'mcp-online' : 'mcp-offline'}`}
                title={srv.error || `Connected via ${srv.transport}`}
              >
                <span className={`dot ${srv.connected ? 'online' : 'offline'}`}></span>
                <span>MCP: {srv.name} ({srv.tools?.length || 0})</span>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-section-title">Saved Conversations</div>

        <div className="threads-list" id="threads-list-container">
          {threads.length === 0 ? (
            <div className="empty-threads">No saved threads yet</div>
          ) : (
            threads.map((tid) => (
              <button
                key={tid}
                className={`thread-item ${activeThreadId === tid ? 'active' : ''}`}
                onClick={() => handleSelectThread(tid)}
                title={tid}
              >
                <span className="thread-icon">💬</span>
                <span className="thread-name">{tid}</span>
              </button>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <div className="status-indicator">
            <span
              className={`status-dot ${
                backendStatus === 'online'
                  ? 'online'
                  : backendStatus === 'offline'
                  ? 'offline'
                  : ''
              }`}
            ></span>
            <span>
              {backendStatus === 'online'
                ? `Online (${mcpToolsCount} MCP tools)`
                : backendStatus === 'offline'
                ? 'Backend Offline'
                : 'Connecting…'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Chat Viewport */}
      <main className="chat-viewport">
        <header className="chat-header">
          <div className="chat-header-info">
            <button
              className="sidebar-toggle-btn"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              aria-label="Toggle sidebar"
            >
              ☰
            </button>
            <h1 className="chat-header-title">Unified LangGraph Assistant</h1>
            <span className="mcp-header-tag">MCP & RAG Enabled</span>
          </div>
          <div className="chat-header-actions">
            <button
              className={`api-key-header-btn ${hasCustomKeys ? 'active-key' : ''}`}
              onClick={() => setShowSettingsModal(true)}
              title="Configure your own OpenAI / Hugging Face API keys"
              id="api-key-settings-button"
            >
              <span className="key-icon">🔑</span>
              <span className="key-label">{hasCustomKeys ? 'Custom Key' : 'API Keys'}</span>
              {hasCustomKeys && <span className="key-pulse-dot" title="Custom user API key is active"></span>}
            </button>

            {activeThreadId ? (
              <div className="thread-tag" title={activeThreadId}>
                Thread: {activeThreadId}
              </div>
            ) : (
              <div className="thread-tag">New Session</div>
            )}
          </div>
        </header>

        {errorMessage && (
          <div className="error-banner">
            <span>⚠️ {errorMessage}</span>
            <button
              className="error-close"
              onClick={() => setErrorMessage(null)}
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        )}

        {successMessage && (
          <div className="success-banner">
            <span>✅ {successMessage}</span>
            <button
              className="error-close"
              onClick={() => setSuccessMessage(null)}
              aria-label="Dismiss message"
            >
              ✕
            </button>
          </div>
        )}

        <div className="messages-container" id="messages-container">
          {isLoadingHistory ? (
            <div className="loading-history">
              <div className="spinner"></div>
              <span>Retrieving thread messages from SQLite...</span>
            </div>
          ) : messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🤖</div>
              <h2 className="empty-state-title">How can I help you today?</h2>
              <p className="empty-state-desc">
                Equipped with native search, calculator, stock prices, <strong>Model Context Protocol (MCP)</strong> servers, and <strong>Persistent PDF Document RAG</strong>.
              </p>
              
              {threadDocuments.length > 0 && (
                <div className="active-doc-banner">
                  <span>📄 Active Document: <strong>{threadDocuments[0].filename}</strong> ({threadDocuments[0].pages} pages indexed)</span>
                </div>
              )}

              <div className="sample-prompts">
                {threadDocuments.length > 0 ? (
                  <button
                    className="prompt-card highlight"
                    onClick={() => handlePromptClick(`What are the key points in ${threadDocuments[0].filename}?`)}
                  >
                    <span>📑 Query Document: What are the key points in {threadDocuments[0].filename}?</span>
                    <span>→</span>
                  </button>
                ) : (
                  <button
                    className="prompt-card"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <span>📄 Click to Upload a PDF to ask questions about your documents</span>
                    <span>→</span>
                  </button>
                )}
                <button
                  className="prompt-card"
                  onClick={() => handlePromptClick("Use the MCP math tool to multiply 42 by 18 and calculate 2 to the power 10.")}
                >
                  <span>🧮 MCP Math: Multiply 42 * 18 and compute 2^10</span>
                  <span>→</span>
                </button>
                <button
                  className="prompt-card"
                  onClick={() => handlePromptClick("What is the current stock price of Apple (AAPL)?")}
                >
                  <span>📈 Native Tool: Check AAPL stock price</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`message-wrapper ${msg.role === 'user' ? 'user' : 'assistant'}`}
              >
                <div className={`avatar ${msg.role === 'user' ? 'user-avatar' : 'assistant-avatar'}`}>
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="bubble">
                  {msg.content}
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="avatar assistant-avatar">🤖</div>
              <div className="typing-bubble">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input Bar */}
        <div className="input-area-container">
          <form className="input-form" onSubmit={handleSendMessage}>
            <input
              ref={inputRef}
              type="text"
              id="chat-message-input"
              className="chat-input"
              placeholder={
                threadDocuments.length > 0
                  ? "Ask anything about your document or use tools..."
                  : "Ask anything, request tool actions, or upload a PDF..."
              }
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={isLoading}
              autoComplete="off"
            />
            <button
              type="submit"
              id="send-message-button"
              className="send-btn"
              disabled={!inputValue.trim() || isLoading}
              aria-label="Send message"
            >
              ➤
            </button>
          </form>
        </div>
      </main>

      {/* Tools & MCP Details Modal */}
      {showToolsModal && (
        <div className="modal-backdrop" onClick={() => setShowToolsModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Registered Tool Ecosystem ({toolsData.total_tools})</h2>
              <button className="modal-close-btn" onClick={() => setShowToolsModal(false)}>✕</button>
            </div>

            <div className="modal-body">
              <h3 className="modal-section-title">Connected MCP Servers</h3>
              <div className="servers-status-grid">
                {serversList.map((srv) => (
                  <div key={srv.name} className={`server-status-card ${srv.connected ? 'active' : 'inactive'}`}>
                    <div className="server-status-head">
                      <strong>{srv.name.toUpperCase()}</strong>
                      <span className={`status-pill ${srv.connected ? 'pill-online' : 'pill-offline'}`}>
                        {srv.connected ? 'Connected' : 'Offline'}
                      </span>
                    </div>
                    <div className="server-status-details">
                      <div>Transport: <code>{srv.transport || 'N/A'}</code></div>
                      <div>Tools: {srv.tools?.length || 0}</div>
                      {srv.error && <div className="server-error-text">Error: {srv.error}</div>}
                    </div>
                  </div>
                ))}
              </div>

              <h3 className="modal-section-title">All Available Tools</h3>
              <div className="tools-list-grid">
                {toolsData.tools?.map((tool) => (
                  <div key={tool.name} className="tool-detail-item">
                    <div className="tool-detail-head">
                      <span className="tool-name"><code>{tool.name}</code></span>
                      <span className={`tool-type-badge ${tool.type}`}>
                        {tool.type === 'mcp' ? `MCP (${tool.server})` : 'Native'}
                      </span>
                    </div>
                    <p className="tool-desc">{tool.description || 'No description provided.'}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* API Key Settings Modal */}
      {showSettingsModal && (
        <div className="modal-backdrop" onClick={() => setShowSettingsModal(false)}>
          <div className="modal-card settings-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-header-title-row">
                <span className="modal-header-icon">🔑</span>
                <h2>API Key & Model Settings</h2>
              </div>
              <button
                className="modal-close-btn"
                onClick={() => setShowSettingsModal(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <div className="settings-active-status-box">
                <span className="status-label">Current Configuration:</span>
                {hasCustomKeys ? (
                  <span className="active-badge custom">
                    <span className="dot online"></span>
                    Custom User Keys Active
                  </span>
                ) : (
                  <span className="active-badge default">
                    <span className="dot offline"></span>
                    Using Server Environment Defaults
                  </span>
                )}
              </div>

              {settingsStatusMessage && (
                <div className="settings-alert-banner">
                  <span>✓ {settingsStatusMessage}</span>
                </div>
              )}

              <form onSubmit={handleSaveApiKeys} className="settings-form">
                {/* OpenAI Section */}
                <div className="settings-field-group">
                  <div className="settings-label-row">
                    <label htmlFor="openai-key-input">OpenAI API Key</label>
                    <span className="badge-pill">ChatOpenAI & RAG Embeddings</span>
                  </div>
                  <div className="password-input-wrapper">
                    <input
                      id="openai-key-input"
                      type={showOpenaiPassword ? 'text' : 'password'}
                      placeholder="sk-proj-... or sk-..."
                      value={openaiKeyInput}
                      onChange={(e) => setOpenaiKeyInput(e.target.value)}
                      className="settings-text-input"
                      autoComplete="off"
                    />
                    <button
                      type="button"
                      className="toggle-visibility-btn"
                      onClick={() => setShowOpenaiPassword(!showOpenaiPassword)}
                      title={showOpenaiPassword ? 'Hide key' : 'Show key'}
                    >
                      {showOpenaiPassword ? '🙈' : '👁️'}
                    </button>
                  </div>
                  <p className="settings-field-hint">
                    Powers default GPT models and computes vector embeddings for your uploaded PDFs.
                  </p>
                </div>

                {/* Hugging Face Section */}
                <div className="settings-field-group">
                  <div className="settings-label-row">
                    <label htmlFor="hf-token-input">Hugging Face Token</label>
                    <span className="badge-pill">ChatHuggingFace</span>
                  </div>
                  <div className="password-input-wrapper">
                    <input
                      id="hf-token-input"
                      type={showHfPassword ? 'text' : 'password'}
                      placeholder="hf_..."
                      value={hfTokenInput}
                      onChange={(e) => setHfTokenInput(e.target.value)}
                      className="settings-text-input"
                      autoComplete="off"
                    />
                    <button
                      type="button"
                      className="toggle-visibility-btn"
                      onClick={() => setShowHfPassword(!showHfPassword)}
                      title={showHfPassword ? 'Hide token' : 'Show token'}
                    >
                      {showHfPassword ? '🙈' : '👁️'}
                    </button>
                  </div>
                  <p className="settings-field-hint">
                    Used for Hugging Face text-generation inference models.
                  </p>
                </div>

                <div className="security-notice-box">
                  <span className="security-icon">🔒</span>
                  <div className="security-text">
                    <strong>Local Client-Side Storage:</strong> Your keys are stored strictly in your browser&apos;s <code>localStorage</code> and transmitted securely per-request to the local backend. They are never saved to database files or logged to disk.
                  </div>
                </div>

                <div className="settings-actions-row">
                  <button
                    type="submit"
                    className="save-settings-btn"
                    id="save-api-keys-btn"
                  >
                    💾 Save Keys
                  </button>
                  <button
                    type="button"
                    className="clear-settings-btn"
                    onClick={handleClearApiKeys}
                    disabled={!openaiKeyInput && !hfTokenInput && !savedOpenaiKey && !savedHfToken}
                  >
                    🗑️ Clear Keys
                  </button>
                  <button
                    type="button"
                    className="cancel-settings-btn"
                    onClick={() => setShowSettingsModal(false)}
                  >
                    Close
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
