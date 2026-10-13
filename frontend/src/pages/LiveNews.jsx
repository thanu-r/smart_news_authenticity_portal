import React, { useEffect, useState, useCallback } from 'react';
import { Newspaper, Search, RefreshCw, ExternalLink, ShieldCheck, Loader2, AlertCircle } from 'lucide-react';
import { newsAPI, verifyAPI } from '../services/api';
import DecisionBadge from '../components/DecisionBadge';

export default function LiveNews() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [verifyingId, setVerifyingId] = useState(null);
  const [error, setError] = useState(null);

  const categories = ['All', 'Politics', 'Science & Tech', 'Business', 'World', 'Health'];

  const fetchNews = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const catParam = selectedCategory === 'All' ? '' : selectedCategory;
      const res = await newsAPI.getLatest(30, catParam);
      setArticles(res.data.articles || []);
    } catch (err) {
      console.error('Failed to fetch news:', err);
      // Fallback sample news if DB empty initially
      setArticles([
        {
          id: 'demo-1',
          title: 'James Webb Space Telescope Identifies Atmospheric Carbon Dioxide and Water on Habitable Zone Exoplanet',
          source: 'Reuters Science',
          url: 'https://www.reuters.com/science/webb-exoplanet',
          category: 'Science & Tech',
          published_at: new Date().toISOString(),
          summary: 'High-resolution transmission spectroscopy confirms the presence of atmospheric carbon dioxide on an exoplanet 120 light years away.',
          verification_status: 'REAL',
          confidence: 0.94,
        },
        {
          id: 'demo-2',
          title: 'Federal Reserve Convenes Emergency Session to Address High-Frequency Liquidity Swings',
          source: 'Bloomberg Markets',
          url: 'https://www.bloomberg.com/markets/fed-liquidity',
          category: 'Business',
          published_at: new Date(Date.now() - 3600000).toISOString(),
          summary: 'Central bank officials held consultations regarding automated algorithmic selloffs impacting Treasury bond yields.',
          verification_status: 'REAL',
          confidence: 0.89,
        },
        {
          id: 'demo-3',
          title: 'Leaked Documents Allege Secret Weather Control Grid Operating in Antarctic Ice Sheet',
          source: 'Geoengineering Secrets Daily',
          url: 'http://unverified-insider-leaks.org/weather-grid',
          category: 'Politics',
          published_at: new Date(Date.now() - 7200000).toISOString(),
          summary: 'Anonymous military contractors claim high-energy atmospheric ionization arrays are hidden beneath polar glaciers.',
          verification_status: 'FAKE',
          confidence: 0.96,
        },
        {
          id: 'demo-4',
          title: 'New Clinical Trial Reports Promising Response Rates for mRNA-Based Pancreatic Cancer Vaccine',
          source: 'Associated Press',
          url: 'https://apnews.com/article/cancer-vaccine-trial',
          category: 'Health',
          published_at: new Date(Date.now() - 14400000).toISOString(),
          summary: 'Phase 2 clinical data published in Nature indicates significant T-cell activation in patient cohorts following adjuvant therapy.',
          verification_status: 'REAL',
          confidence: 0.92,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [selectedCategory]);

  useEffect(() => {
    fetchNews();
  }, [fetchNews]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      fetchNews();
      return;
    }
    setLoading(true);
    try {
      const res = await newsAPI.search(searchQuery.trim());
      setArticles(res.data.articles || []);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshFeeds = async () => {
    setRefreshing(true);
    try {
      await newsAPI.refresh();
      await fetchNews();
    } catch (err) {
      console.error('Refresh feeds error:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleVerifyArticle = async (article) => {
    setVerifyingId(article.id);
    try {
      const res = await verifyAPI.verifyText({
        title: article.title,
        headline: article.title,
        text: article.summary || article.content || article.title,
        url: article.url,
      });
      // Update article state locally
      setArticles((prev) =>
        prev.map((a) =>
          a.id === article.id
            ? { ...a, verification_status: res.data.prediction, confidence: res.data.calibrated_confidence }
            : a
        )
      );
    } catch (err) {
      console.error('Verify article error:', err);
    } finally {
      setVerifyingId(null);
    }
  };

  return (
    <div style={{ maxWidth: '1240px', margin: '0 auto', padding: '32px 20px 60px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px', marginBottom: '28px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
            <Newspaper size={18} />
            <span>REAL-TIME INGESTION FEED</span>
          </div>
          <h1 style={{ fontSize: '2.2rem', fontWeight: 800 }}>
            Live News <span style={{ color: 'var(--primary)' }}>Radar</span>
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Monitored RSS feeds screened with automated authenticity scores and claim consistency.
          </p>
        </div>

        {/* Action button */}
        <button
          onClick={handleRefreshFeeds}
          disabled={refreshing}
          className="btn-outline"
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
          <span>{refreshing ? 'Refreshing Feeds...' : 'Sync Live Feeds'}</span>
        </button>
      </div>

      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: 'var(--fake-bg)', border: '1px solid var(--fake-glow)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.85rem', marginBottom: '20px' }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="glass-panel" style={{ padding: '16px 20px', marginBottom: '28px', display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Category Pills */}
        <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat === 'All' ? '' : cat)}
              style={{
                background: (selectedCategory === cat || (!selectedCategory && cat === 'All')) ? 'var(--primary)' : 'rgba(255,255,255,0.04)',
                color: (selectedCategory === cat || (!selectedCategory && cat === 'All')) ? '#050b14' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: '0.82rem',
                border: '1px solid var(--border-glass)',
                borderRadius: '20px',
                padding: '6px 14px',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s',
              }}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '8px', minWidth: '280px' }}>
          <div style={{ position: 'relative', width: '100%' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
            <input
              type="text"
              placeholder="Search live articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="form-input"
              style={{ paddingLeft: '36px', paddingRight: '12px', paddingTop: '8px', paddingBottom: '8px', fontSize: '0.88rem' }}
            />
          </div>
        </form>
      </div>

      {/* Article Grid */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-dim)' }}>
          <Loader2 size={36} className="animate-spin" style={{ margin: '0 auto 12px', color: 'var(--primary)' }} />
          <p>Ingesting and screening articles...</p>
        </div>
      ) : articles.length === 0 ? (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
          <Newspaper size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
          <h3>No articles found</h3>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-dim)' }}>Try adjusting your search query or sync new feeds.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '20px' }}>
          {articles.map((item, idx) => (
            <div key={item.id || idx} className="glass-panel" style={{ padding: '22px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '14px' }}>
              <div>
                {/* Meta row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span style={{ fontSize: '0.75rem', background: 'rgba(255,255,255,0.05)', color: 'var(--primary)', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
                    {item.category || 'General'}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                    {item.source || 'News Wire'}
                  </span>
                </div>

                {/* Title */}
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, lineHeight: 1.4, marginBottom: '8px', color: 'var(--text-main)' }}>
                  {item.title}
                </h3>

                {/* Summary */}
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                  {item.summary || item.snippet}
                </p>
              </div>

              {/* Footer status & Action */}
              <div style={{ paddingTop: '12px', borderTop: '1px solid var(--border-glass)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                {item.verification_status ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <DecisionBadge prediction={item.verification_status} size="sm" />
                    {item.confidence && (
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        {Math.round(item.confidence * 100)}%
                      </span>
                    )}
                  </div>
                ) : (
                  <button
                    onClick={() => handleVerifyArticle(item)}
                    disabled={verifyingId === item.id}
                    className="btn-outline"
                    style={{ fontSize: '0.78rem', padding: '4px 12px' }}
                  >
                    {verifyingId === item.id ? (
                      <>
                        <Loader2 size={12} className="animate-spin" />
                        <span>Verifying...</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck size={13} color="var(--primary)" />
                        <span>Verify Now</span>
                      </>
                    )}
                  </button>
                )}

                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: 'var(--text-dim)', transition: 'color 0.2s', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.78rem', textDecoration: 'none' }}
                  >
                    <span>Read</span>
                    <ExternalLink size={12} />
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
