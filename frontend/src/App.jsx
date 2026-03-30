import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { queryNews, getChatHistory, deleteChat, clearChatHistory } from './api';

const formatDuration = (value) => {
  if (value == null) return '--';
  if (value < 1000) return `${value.toFixed(1)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
};

const TIMING_LABELS = {
  extract_entities_ms: 'Entity',
  expand_query_ms: 'Expand',
  search_ms: 'Search',
  rerank_ms: 'Rerank',
  entity_boost_ms: 'Boost',
  answer_ms: 'LLM',
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
    loadChatHistory();
  }, []);

  const loadChatHistory = async () => {
    setHistoryLoading(true);
    try {
      const data = await getChatHistory(50);
      setChatHistory(data.chats || []);
    } catch (err) {
      console.error('Failed to load chat history:', err);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const result = await queryNews(userMessage);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: result.markdown_response,
          timing: result.timing,
          clientMs: result.client_ms,
        },
      ]);
      // Refresh chat history after new query
      loadChatHistory();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `**Error:** ${err.response?.data?.detail || err.message}`,
        },
      ]);
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
    // Reconstruct the conversation from history
    const newMessages = [
      { role: 'user', content: chat.query },
    ];
    
    // Use stored markdown_response if available, otherwise build from data
    let markdown = chat.markdown_response;
    if (!markdown) {
      // Fallback: Build markdown response from stored data
      markdown = `# Query Results: "${chat.query}"\n\n`;
      if (chat.explanation) {
        markdown += `## Summary\n${chat.explanation}\n\n`;
      }
      markdown += `## ${chat.stories_count} source${chat.stories_count !== 1 ? 's' : ''} found.\n\n`;
      chat.stories?.forEach((story, i) => {
        markdown += `### ${i + 1}. ${story.title}\n\n`;
        markdown += `**Source:** ${story.source || 'Unknown'}\n\n`;
      });
    }
    
    newMessages.push({
      role: 'assistant',
      content: markdown,
      timing: chat.timing || null,
    });
    setMessages(newMessages);
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

  const suggestions = [
    'HDFC Bank news',
    'Banking sector update',
    'RBI policy changes',
    'Interest rate impact',
  ];

  return (
    <div className="app-layout">
      {/* Sidebar */}
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
                      {chat.stories_count} sources • {formatDate(chat.created_at)}
                    </span>
                  </div>
                  <button
                    className="history-delete"
                    onClick={(e) => handleDeleteChat(e, chat.id)}
                    title="Delete chat"
                  >
                    ×
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

      {/* Toggle button when sidebar is closed */}
      {!sidebarOpen && (
        <button className="sidebar-open-btn" onClick={() => setSidebarOpen(true)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      )}

      {/* Main Chat Area */}
      <div className="chat-container">
        {/* Header */}
        <header className="chat-header">
          <h1>Financial News Intelligence</h1>
        </header>

        {/* Messages Area */}
        <main className="chat-messages">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">📊</div>
              <h2>How can I help you today?</h2>
              <p>Ask me about financial news, market updates, or company information.</p>
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
                    {msg.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="message-content">
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
                  <div className="message-avatar">🤖</div>
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

        {/* Input Area */}
        <footer className="chat-input-area">
          <form onSubmit={handleSubmit} className="chat-form">
            <div className="input-wrapper">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about financial news..."
                disabled={loading}
              />
              <button
                type="submit"
                disabled={!input.trim() || loading}
                className="send-btn"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </button>
            </div>
          </form>
          <p className="disclaimer">
            AI-powered financial news analysis. Results may vary.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
