import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  clearChatHistory,
  deleteChat,
  getChatHistory,
  getCurrentUser,
  getRecommendations,
  loginUser,
  logoutUser,
  queryNews,
  queryWithAttachments,
  registerUser,
} from './api';

const formatDuration = (value) => {
  if (value == null) return '--';
  if (value < 1000) return `${value.toFixed(1)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
};

const TIMING_LABELS = {
  context_build_ms: 'Context',
  intent_classify_ms: 'Intent',
  extract_entities_ms: 'Entity',
  expand_query_ms: 'Expand',
  search_ms: 'Search',
  rerank_ms: 'Rerank',
  entity_boost_ms: 'Boost',
  attachment_rank_ms: 'Attach',
  general_llm_ms: 'LLM',
  refresh_feeds_ms: 'Refresh',
  refresh_persist_ms: 'Persist',
  refresh_dataset_scan_ms: 'Sync Scan',
  refresh_ingest_ms: 'Ingest',
  answer_ms: 'LLM',
};

const INTENT_LABELS = {
  general_chat: 'General',
  news_update: 'Refresh',
  financial_query: 'Finance',
};

const suggestions = [
  '泡泡玛特的股票能买吗？',
  '更新一下新闻',
  '最近港股消费板块怎么样？',
  '帮我解释一下什么是通货膨胀',
];

const emptyRegisterForm = {
  username: '',
  email: '',
  display_name: '',
  password: '',
};

const emptyLoginForm = {
  identifier: '',
  password: '',
};

function App() {
  const [activeView, setActiveView] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendations, setRecommendations] = useState({ mode: 'latest', feed_summary: '', cards: [] });
  const [sessionLoading, setSessionLoading] = useState(true);
  const [currentUser, setCurrentUser] = useState(null);
  const [authMode, setAuthMode] = useState('login');
  const [loginForm, setLoginForm] = useState(emptyLoginForm);
  const [registerForm, setRegisterForm] = useState(emptyRegisterForm);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    bootstrapSession();
  }, []);

  useEffect(() => {
    if (currentUser) {
      inputRef.current?.focus();
    }
  }, [currentUser]);

  const bootstrapSession = async () => {
    setSessionLoading(true);
    try {
      const user = await getCurrentUser();
      setCurrentUser(user);
      await Promise.all([loadChatHistory(), loadRecommendations()]);
    } catch (err) {
      setCurrentUser(null);
      setChatHistory([]);
      setRecommendations({ mode: 'latest', feed_summary: '', cards: [] });
    } finally {
      setSessionLoading(false);
    }
  };

  const loadChatHistory = async () => {
    setHistoryLoading(true);
    try {
      const data = await getChatHistory(50);
      setChatHistory(data.chats || []);
    } catch (err) {
      if (err.response?.status === 401) {
        setCurrentUser(null);
        setChatHistory([]);
        setMessages([]);
      } else {
        console.error('Failed to load chat history:', err);
      }
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadRecommendations = async () => {
    setRecommendationLoading(true);
    try {
      const data = await getRecommendations();
      setRecommendations({
        mode: data.mode || 'latest',
        feed_summary: data.feed_summary || '',
        cards: data.cards || [],
      });
    } catch (err) {
      if (err.response?.status === 401) {
        setCurrentUser(null);
        setChatHistory([]);
        setMessages([]);
        setRecommendations({ mode: 'latest', feed_summary: '', cards: [] });
      } else {
        console.error('Failed to load recommendations:', err);
      }
    } finally {
      setRecommendationLoading(false);
    }
  };

  const buildHistoryPayload = (historyMessages) => {
    return historyMessages.slice(-6).map((message) => ({
      role: message.role,
      content: message.content,
      intent: message.intent || null,
      matched_entities: Array.isArray(message.matchedEntities)
        ? message.matchedEntities
            .map((entity) => (typeof entity === 'string' ? entity : entity?.name))
            .filter(Boolean)
        : [],
      story_titles: Array.isArray(message.stories)
        ? message.stories
            .map((story) => story?.title)
            .filter(Boolean)
            .slice(0, 3)
        : [],
    }));
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    if (!loginForm.identifier.trim() || !loginForm.password) return;
    setAuthLoading(true);
    setAuthError('');
    try {
      const user = await loginUser({
        identifier: loginForm.identifier.trim(),
        password: loginForm.password,
      });
      setCurrentUser(user);
      setLoginForm(emptyLoginForm);
      setMessages([]);
      await Promise.all([loadChatHistory(), loadRecommendations()]);
    } catch (err) {
      setAuthError(err.response?.data?.detail || 'Login failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    if (!registerForm.username.trim() || !registerForm.email.trim() || !registerForm.password) return;
    setAuthLoading(true);
    setAuthError('');
    try {
      const user = await registerUser({
        username: registerForm.username.trim(),
        email: registerForm.email.trim(),
        display_name: registerForm.display_name.trim() || undefined,
        password: registerForm.password,
      });
      setCurrentUser(user);
      setRegisterForm(emptyRegisterForm);
      setMessages([]);
      await Promise.all([loadChatHistory(), loadRecommendations()]);
    } catch (err) {
      setAuthError(err.response?.data?.detail || 'Registration failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logoutUser();
    } catch (err) {
      console.error('Failed to logout:', err);
    } finally {
      setCurrentUser(null);
      setMessages([]);
      setChatHistory([]);
      setRecommendations({ mode: 'latest', feed_summary: '', cards: [] });
      setSelectedFile(null);
      setFileError('');
      setAuthError('');
      setLoginForm(emptyLoginForm);
      setRegisterForm(emptyRegisterForm);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!currentUser || !input.trim() || loading) return;

    const userMessage = input.trim();
    const pendingFile = selectedFile;
    const historyPayload = buildHistoryPayload(messages);
    setFileError('');
    setInput('');
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: userMessage,
        attachmentName: pendingFile?.name || null,
      },
    ]);
    setLoading(true);

    try {
      const result = pendingFile
        ? await queryWithAttachments(userMessage, historyPayload, pendingFile)
        : await queryNews(userMessage, historyPayload);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: result.markdown_response,
          timing: result.timing,
          clientMs: result.client_ms,
          intent: result.intent,
          intentSource: result.intent_source,
          matchedEntities: result.matched_entities || [],
          stories: result.stories || [],
          attachmentSummary: result.attachment_summary || '',
          attachmentEvidence: result.attachment_evidence || [],
        },
      ]);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      await Promise.all([loadChatHistory(), loadRecommendations()]);
    } catch (err) {
      if (err.response?.status === 401) {
        setCurrentUser(null);
        setChatHistory([]);
        setMessages([]);
        setAuthError('Session expired. Please log in again.');
      } else {
        if (pendingFile) {
          setFileError(err.response?.data?.detail || 'Attachment upload failed');
        }
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `**Error:** ${err.response?.data?.detail || err.message}`,
          },
        ]);
      }
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleAttachmentPick = () => {
    if (!loading) {
      fileInputRef.current?.click();
    }
  };

  const handleAttachmentChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const lowerName = file.name.toLowerCase();
    const allowed = ['.pdf', '.png', '.jpg', '.jpeg'];
    const isAllowed = allowed.some((suffix) => lowerName.endsWith(suffix));
    if (!isAllowed) {
      setFileError('仅支持 PDF、PNG、JPG、JPEG');
      e.target.value = '';
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setFileError('附件大小不能超过 10 MB');
      e.target.value = '';
      return;
    }
    setFileError('');
    setSelectedFile(file);
  };

  const clearAttachment = () => {
    setSelectedFile(null);
    setFileError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleNewChat = () => {
    setActiveView('chat');
    setMessages([]);
    clearAttachment();
    inputRef.current?.focus();
  };

  const handleSelectChat = (chat) => {
    setActiveView('chat');
    const reconstructed = [
      {
        role: 'user',
        content: chat.query,
        attachmentName: chat.attachments?.[0]?.file_name || null,
      },
    ];

    let markdown = chat.markdown_response;
    if (!markdown) {
      const heading = chat.intent === 'general_chat'
        ? 'General Answer'
        : chat.intent === 'news_update'
          ? 'News Refresh'
          : 'Query Results';
      markdown = `# ${heading}: "${chat.query}"\n\n`;
      if (chat.explanation) {
        markdown += `## Summary\n${chat.explanation}\n\n`;
      }
      markdown += `## ${chat.stories_count} source${chat.stories_count !== 1 ? 's' : ''} found.\n\n`;
      chat.stories?.forEach((story, i) => {
        markdown += `### ${i + 1}. ${story.title}\n\n`;
        markdown += `**Source:** ${story.source || 'Unknown'}\n\n`;
      });
    }

    reconstructed.push({
      role: 'assistant',
      content: markdown,
      timing: chat.timing || null,
      intent: chat.intent || 'financial_query',
      intentSource: chat.intent_source || 'pipeline',
      matchedEntities: chat.matched_entities || [],
      stories: chat.stories || [],
    });
    setMessages(reconstructed);
  };

  const handleDeleteChat = async (e, chatId) => {
    e.stopPropagation();
    try {
      await deleteChat(chatId);
      setChatHistory((prev) => prev.filter((c) => c.id !== chatId));
      await loadRecommendations();
    } catch (err) {
      console.error('Failed to delete chat:', err);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all chat history?')) return;
    try {
      await clearChatHistory();
      setChatHistory([]);
      await loadRecommendations();
    } catch (err) {
      console.error('Failed to clear history:', err);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'Unknown date';
    const date = new Date(dateStr);
    if (Number.isNaN(date.getTime())) return dateStr;
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  const renderRecommendations = () => {
    const modeLabel = recommendations.mode === 'personalized' ? '与你最近关注相关' : '最新 10 篇资讯';

    return (
      <section className="recommendation-section">
        <div className="recommendation-header">
          <div>
            <p className="recommendation-kicker">Daily Cards</p>
            <h3>{modeLabel}</h3>
          </div>
          {recommendationLoading ? <span className="recommendation-status">Loading...</span> : null}
        </div>
        <p className="recommendation-summary">
          {recommendations.feed_summary || '根据你的最近互动整理卡片。'}
        </p>
        {recommendations.cards?.length ? (
          <div className="recommendation-cards">
            {recommendations.cards.map((card) => (
              <article key={card.story_id} className="recommendation-card">
                <div className="recommendation-card-top">
                  <span>{card.source || 'Unknown source'}</span>
                  <span>{formatDate(card.published_date)}</span>
                </div>
                <h4>{card.title}</h4>
                <p>{card.summary}</p>
                {card.matched_entities?.length ? (
                  <div className="recommendation-tags">
                    {card.matched_entities.map((entity) => (
                      <span key={`${card.story_id}-${entity}`} className="recommendation-tag">
                        {entity}
                      </span>
                    ))}
                  </div>
                ) : null}
                {card.stock_symbols?.length ? (
                  <div className="recommendation-stocks">
                    {card.stock_symbols.map((symbol) => (
                      <span key={`${card.story_id}-${symbol}`} className="recommendation-stock">
                        {symbol}
                      </span>
                    ))}
                  </div>
                ) : null}
                <div className="recommendation-footer">
                  <strong>{card.recommendation_label}</strong>
                  <span>{card.recommendation_reason}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="recommendation-empty">
            Start chatting about companies, sectors, or regulators and recommendation cards will appear here.
          </div>
        )}
      </section>
    );
  };

  const renderWelcome = () => (
    <div className="welcome">
      <div className="welcome-icon">FN</div>
      <h2>Ask about markets, sectors, and company news.</h2>
      <p>Your account is active. Query results are saved only to your own history.</p>
      <div className="suggestions">
        {suggestions.map((s) => (
          <button
            key={s}
            className="suggestion"
            onClick={() => {
              setActiveView('chat');
              setInput(s);
            }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );

  if (sessionLoading) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-loading">Loading session...</div>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <div className="auth-header">
            <p className="auth-kicker">Financial News Intelligence</p>
            <h1>Account Access</h1>
            <p className="auth-copy">
              Register once to get an isolated private knowledge space and access the shared public knowledge base.
            </p>
          </div>

          <div className="auth-tabs">
            <button
              className={`auth-tab ${authMode === 'login' ? 'active' : ''}`}
              onClick={() => {
                setAuthMode('login');
                setAuthError('');
              }}
              type="button"
            >
              Login
            </button>
            <button
              className={`auth-tab ${authMode === 'register' ? 'active' : ''}`}
              onClick={() => {
                setAuthMode('register');
                setAuthError('');
              }}
              type="button"
            >
              Register
            </button>
          </div>

          {authError ? <div className="auth-error">{authError}</div> : null}

          {authMode === 'login' ? (
            <form className="auth-form" onSubmit={handleLoginSubmit}>
              <label>
                Username or email
                <input
                  type="text"
                  value={loginForm.identifier}
                  onChange={(e) => setLoginForm((prev) => ({ ...prev, identifier: e.target.value }))}
                  placeholder="yourname or you@example.com"
                  autoComplete="username"
                  disabled={authLoading}
                />
              </label>
              <label>
                Password
                <input
                  type="password"
                  value={loginForm.password}
                  onChange={(e) => setLoginForm((prev) => ({ ...prev, password: e.target.value }))}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  disabled={authLoading}
                />
              </label>
              <button className="auth-submit" type="submit" disabled={authLoading}>
                {authLoading ? 'Signing in...' : 'Sign In'}
              </button>
            </form>
          ) : (
            <form className="auth-form" onSubmit={handleRegisterSubmit}>
              <label>
                Username
                <input
                  type="text"
                  value={registerForm.username}
                  onChange={(e) => setRegisterForm((prev) => ({ ...prev, username: e.target.value }))}
                  placeholder="Choose a username"
                  autoComplete="username"
                  disabled={authLoading}
                />
              </label>
              <label>
                Email
                <input
                  type="email"
                  value={registerForm.email}
                  onChange={(e) => setRegisterForm((prev) => ({ ...prev, email: e.target.value }))}
                  placeholder="you@example.com"
                  autoComplete="email"
                  disabled={authLoading}
                />
              </label>
              <label>
                Display name
                <input
                  type="text"
                  value={registerForm.display_name}
                  onChange={(e) => setRegisterForm((prev) => ({ ...prev, display_name: e.target.value }))}
                  placeholder="Optional display name"
                  autoComplete="name"
                  disabled={authLoading}
                />
              </label>
              <label>
                Password
                <input
                  type="password"
                  value={registerForm.password}
                  onChange={(e) => setRegisterForm((prev) => ({ ...prev, password: e.target.value }))}
                  placeholder="Create a password"
                  autoComplete="new-password"
                  disabled={authLoading}
                />
              </label>
              <button className="auth-submit" type="submit" disabled={authLoading}>
                {authLoading ? 'Creating account...' : 'Create Account'}
              </button>
            </form>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app-layout">
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={handleNewChat}>
            <span>+</span> New Chat
          </button>
          <button className="toggle-sidebar" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        <div className="sidebar-user">
          <div className="sidebar-user-meta">
            <strong>{currentUser.display_name || currentUser.username}</strong>
            <span>{currentUser.email}</span>
            <small>
              Public: {currentUser.public_namespace?.slug}
              {currentUser.default_private_namespace?.slug ? ` | Private: ${currentUser.default_private_namespace.slug}` : ''}
            </small>
          </div>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>

        <div className="sidebar-nav">
          <button
            className={`sidebar-nav-item ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveView('chat')}
            type="button"
          >
            Chat
          </button>
          <button
            className={`sidebar-nav-item ${activeView === 'recommendations' ? 'active' : ''}`}
            onClick={() => setActiveView('recommendations')}
            type="button"
          >
            Daily Recommendations
          </button>
        </div>

        <div className="chat-history">
          {historyLoading ? (
            <div className="history-loading">Loading...</div>
          ) : chatHistory.length === 0 ? (
            <div className="history-empty">No chat history</div>
          ) : (
            <div className="history-list">
              {chatHistory.map((chat) => (
                <div
                  key={chat.id}
                  className="history-item"
                  onClick={() => handleSelectChat(chat)}
                >
                  <div className="history-item-content">
                    <span className="history-query">{chat.query}</span>
                    <span className="history-meta">
                      {(INTENT_LABELS[chat.intent] || 'Finance')} • {chat.stories_count} sources • {formatDate(chat.created_at)}
                    </span>
                  </div>
                  <button
                    className="history-delete"
                    onClick={(e) => handleDeleteChat(e, chat.id)}
                    title="Delete chat"
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {chatHistory.length > 0 && (
          <div className="sidebar-footer">
            <button className="clear-history-btn" onClick={handleClearHistory}>
              Clear History
            </button>
          </div>
        )}
      </aside>

      {!sidebarOpen && (
        <button className="sidebar-open-btn" onClick={() => setSidebarOpen(true)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      )}

      <div className="chat-container">
        <header className="chat-header">
          <div>
            <h1>Financial News Intelligence</h1>
            <p className="header-subtitle">Authenticated access to public knowledge with per-user history isolation.</p>
          </div>
        </header>

        <main className="chat-messages">
          {activeView === 'recommendations' ? (
            <div className="dashboard dashboard-recommendations">
              {renderRecommendations()}
            </div>
          ) : messages.length === 0 ? (
            <div className="dashboard">
              {renderWelcome()}
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div className="message-content">
                    {msg.role === 'assistant' && msg.intent && (
                      <div className="intent-chip">
                        Route: {INTENT_LABELS[msg.intent] || msg.intent}
                        {msg.intentSource ? ` · ${msg.intentSource}` : ''}
                      </div>
                    )}
                    {msg.role === 'assistant' && (msg.timing || msg.clientMs != null) && (
                      <div className="timing-card">
                        <div className="timing-summary">
                          <span>Browser: {formatDuration(msg.clientMs)}</span>
                          <span>Pipeline: {formatDuration(msg.timing?.pipeline_ms)}</span>
                          <span>API: {formatDuration(msg.timing?.api_ms)}</span>
                        </div>
                        {msg.timing?.stages && Object.keys(msg.timing.stages).length > 0 && (
                          <div className="timing-stages">
                            {Object.entries(msg.timing.stages).map(([key, value]) => (
                              <span key={key} className="timing-stage">
                                {TIMING_LABELS[key] || key}: {formatDuration(value)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    {msg.role === 'user' ? (
                      <div>
                        <p>{msg.content}</p>
                        {msg.attachmentName && <p>[附件] {msg.attachmentName}</p>}
                      </div>
                    ) : (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message assistant">
                  <div className="message-avatar">AI</div>
                  <div className="message-content">
                    <div className="typing">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {activeView === 'chat' ? (
          <footer className="chat-input-area">
            <form onSubmit={handleSubmit} className="chat-form">
              <div className="input-wrapper">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  onChange={handleAttachmentChange}
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  onClick={handleAttachmentPick}
                  disabled={loading}
                  className="send-btn"
                  title="上传 PDF 或图片作为当前问题附件"
                >
                  +
                </button>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入通识问题、更新新闻，或直接问金融问题..."
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || loading}
                  className="send-btn"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                  </svg>
                </button>
              </div>
              {(selectedFile || fileError) && (
                <div>
                  {selectedFile && (
                    <p>
                      当前附件: {selectedFile.name}
                      {' '}
                      <button type="button" onClick={clearAttachment} disabled={loading}>
                        移除
                      </button>
                    </p>
                  )}
                  {fileError && <p>{fileError}</p>}
                </div>
              )}
            </form>
            <p className="disclaimer">
              Public knowledge base is shared. Your private namespace metadata and chat history stay isolated to your account.
            </p>
          </footer>
        ) : null}
      </div>
    </div>
  );
}

export default App;
