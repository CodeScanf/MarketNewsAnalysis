import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  clearChatHistory,
  deleteChat,
  getChatHistory,
  getCurrentUser,
  loginUser,
  logoutUser,
  queryNews,
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
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(true);
  const [currentUser, setCurrentUser] = useState(null);
  const [authMode, setAuthMode] = useState('login');
  const [loginForm, setLoginForm] = useState(emptyLoginForm);
  const [registerForm, setRegisterForm] = useState(emptyRegisterForm);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

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
      await loadChatHistory();
    } catch (err) {
      setCurrentUser(null);
      setChatHistory([]);
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
      await loadChatHistory();
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
      await loadChatHistory();
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
      setAuthError('');
      setLoginForm(emptyLoginForm);
      setRegisterForm(emptyRegisterForm);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!currentUser || !input.trim() || loading) return;

    const userMessage = input.trim();
    const historyPayload = buildHistoryPayload(messages);
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const result = await queryNews(userMessage, historyPayload);
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
        },
      ]);
      await loadChatHistory();
    } catch (err) {
      if (err.response?.status === 401) {
        setCurrentUser(null);
        setChatHistory([]);
        setMessages([]);
        setAuthError('Session expired. Please log in again.');
      } else {
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

  const handleNewChat = () => {
    setMessages([]);
    inputRef.current?.focus();
  };

  const handleSelectChat = (chat) => {
    const reconstructed = [{ role: 'user', content: chat.query }];

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
    } catch (err) {
      console.error('Failed to delete chat:', err);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all chat history?')) return;
    try {
      await clearChatHistory();
      setChatHistory([]);
    } catch (err) {
      console.error('Failed to clear history:', err);
    }
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

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
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">FN</div>
              <h2>Ask about markets, sectors, and company news.</h2>
              <p>Your account is active. Query results are saved only to your own history.</p>
              <div className="suggestions">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    className="suggestion"
                    onClick={() => setInput(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
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
                      <p>{msg.content}</p>
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

        <footer className="chat-input-area">
          <form onSubmit={handleSubmit} className="chat-form">
            <div className="input-wrapper">
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
          </form>
          <p className="disclaimer">
            Public knowledge base is shared. Your private namespace metadata and chat history stay isolated to your account.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
