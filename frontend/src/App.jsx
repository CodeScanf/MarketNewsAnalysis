import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { queryNews, ingestArticles, runDemo, getStats } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('query');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [markdown, setMarkdown] = useState('');
  const [stats, setStats] = useState(null);

  // Query state
  const [query, setQuery] = useState('');

  // Ingest state
  const [articles, setArticles] = useState([]);
  const [newArticle, setNewArticle] = useState({ title: '', content: '', source: '' });

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const result = await queryNews(query);
      setMarkdown(result.markdown_response);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickQuery = (q) => {
    setQuery(q);
  };

  const handleDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await runDemo();
      setMarkdown(result.markdown_response);
      fetchStats();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const addArticle = () => {
    if (!newArticle.title.trim() || !newArticle.content.trim()) return;
    setArticles([...articles, { ...newArticle }]);
    setNewArticle({ title: '', content: '', source: '' });
  };

  const removeArticle = (index) => {
    setArticles(articles.filter((_, i) => i !== index));
  };

  const handleIngest = async () => {
    if (articles.length === 0) return;

    setLoading(true);
    setError(null);
    try {
      const result = await ingestArticles(articles);
      setMarkdown(`# Ingestion Complete\n\n**Total Articles:** ${result.total_articles}\n\n**Unique Stories:** ${result.unique_count}\n\n**Duplicates Found:** ${result.duplicate_count}\n\n**Skipped:** ${result.skipped_count}\n\n${result.message}`);
      setArticles([]);
      fetchStats();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🏦 Financial News Intelligence</h1>
        <p>AI-Powered news processing, deduplication, and intelligent querying</p>
      </header>

      {stats && (
        <div className="stats">
          <div className="stat">
            <div className="stat-value">{stats.total_stories}</div>
            <div className="stat-label">Total Stories</div>
          </div>
          <div className="stat">
            <div className="stat-value">{stats.indexed_stories}</div>
            <div className="stat-label">Indexed</div>
          </div>
        </div>
      )}

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'query' ? 'active' : ''}`}
          onClick={() => setActiveTab('query')}
        >
          🔍 Query
        </button>
        <button
          className={`tab ${activeTab === 'ingest' ? 'active' : ''}`}
          onClick={() => setActiveTab('ingest')}
        >
          📥 Ingest
        </button>
        <button
          className={`tab ${activeTab === 'demo' ? 'active' : ''}`}
          onClick={() => setActiveTab('demo')}
        >
          🎮 Demo
        </button>
      </div>

      {error && <div className="error">⚠️ {error}</div>}

      {activeTab === 'query' && (
        <div className="card">
          <h2>Query News</h2>
          <form onSubmit={handleQuery}>
            <div className="form-group">
              <label>Enter your query</label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., HDFC Bank news, Banking sector update, RBI policy changes"
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>

          <div className="quick-actions">
            <span style={{ color: '#71717a', fontSize: '0.8rem' }}>Quick queries:</span>
            <button className="quick-action" onClick={() => handleQuickQuery('HDFC Bank news')}>
              HDFC Bank
            </button>
            <button className="quick-action" onClick={() => handleQuickQuery('Banking sector update')}>
              Banking Sector
            </button>
            <button className="quick-action" onClick={() => handleQuickQuery('RBI policy changes')}>
              RBI Policy
            </button>
            <button className="quick-action" onClick={() => handleQuickQuery('Interest rate impact')}>
              Interest Rates
            </button>
          </div>
        </div>
      )}

      {activeTab === 'ingest' && (
        <div className="card">
          <h2>Ingest Articles</h2>
          
          <div className="form-group">
            <label>Article Title</label>
            <input
              type="text"
              value={newArticle.title}
              onChange={(e) => setNewArticle({ ...newArticle, title: e.target.value })}
              placeholder="Enter article title"
            />
          </div>
          
          <div className="form-group">
            <label>Content</label>
            <textarea
              value={newArticle.content}
              onChange={(e) => setNewArticle({ ...newArticle, content: e.target.value })}
              placeholder="Enter article content"
            />
          </div>
          
          <div className="form-group">
            <label>Source (optional)</label>
            <input
              type="text"
              value={newArticle.source}
              onChange={(e) => setNewArticle({ ...newArticle, source: e.target.value })}
              placeholder="e.g., Economic Times"
            />
          </div>
          
          <button className="btn btn-secondary" onClick={addArticle} style={{ marginRight: '0.5rem' }}>
            Add Article
          </button>
          
          {articles.length > 0 && (
            <>
              <h3 style={{ margin: '1.5rem 0 1rem', fontSize: '0.95rem' }}>
                Articles to Ingest ({articles.length})
              </h3>
              {articles.map((article, index) => (
                <div key={index} className="article-item">
                  <h4>{article.title}</h4>
                  <p>{article.content.substring(0, 100)}...</p>
                  <button className="remove-btn" onClick={() => removeArticle(index)}>
                    Remove
                  </button>
                </div>
              ))}
              <button className="btn btn-primary" onClick={handleIngest} disabled={loading} style={{ marginTop: '1rem' }}>
                {loading ? 'Processing...' : `Ingest ${articles.length} Article(s)`}
              </button>
            </>
          )}
        </div>
      )}

      {activeTab === 'demo' && (
        <div className="card">
          <h2>Run Demo</h2>
          <p style={{ color: '#a1a1aa', marginBottom: '1rem' }}>
            Load sample articles from the problem statement and process them through the intelligence system.
          </p>
          <button className="btn btn-primary" onClick={handleDemo} disabled={loading}>
            {loading ? 'Running Demo...' : 'Run Demo'}
          </button>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          Processing...
        </div>
      )}

      {markdown && !loading && (
        <div className="card">
          <h2>Results</h2>
          <div className="markdown-content">
            <ReactMarkdown>{markdown}</ReactMarkdown>
          </div>
        </div>
      )}

      {!markdown && !loading && (
        <div className="empty-state">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
          </svg>
          <p>Run a query or demo to see results</p>
        </div>
      )}
    </div>
  );
}

export default App;
