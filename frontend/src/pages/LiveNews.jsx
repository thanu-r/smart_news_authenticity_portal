import React, { useEffect, useState, useCallback } from 'react';
import {
  Newspaper, Search, RefreshCw, ExternalLink, ShieldCheck,
  Loader2, AlertCircle, ChevronDown, ChevronUp, CheckCircle,
  XCircle, HelpCircle, Calendar, Clock, Globe
} from 'lucide-react';
import { newsAPI, verifyAPI } from '../services/api';
import DecisionBadge from '../components/DecisionBadge';

export default function LiveNews() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [verifyingId, setVerifyingId] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [error, setError] = useState(null);

  const categories = ['All', 'Politics', 'Science & Tech', 'Business', 'World', 'Health', 'General'];

  const fetchNews = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const catParam = selectedCategory === 'All' ? '' : selectedCategory;
      const res = await newsAPI.getLatest(30, catParam);
      setArticles(res.data.articles || []);
    } catch (err) {
      console.error('Failed to fetch news:', err);
      setError('Unable to load live news from server. Please sync live feeds.');
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
      setError('Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshFeeds = async () => {
    setRefreshing(true);
    setError(null);
    try {
      await newsAPI.refresh();
      await fetchNews();
    } catch (err) {
      console.error('Refresh feeds error:', err);
      setError('Failed to refresh RSS feeds. Check network connectivity.');
    } finally {
      setRefreshing(false);
    }
  };

  const handleVerifyArticle = async (article) => {
    setVerifyingId(article.id);
    setError(null);
    try {
      const res = await newsAPI.verifyOne(article.id);
      const updatedItem = res.data;
      setArticles((prev) =>
        prev.map((a) =>
          a.id === article.id
            ? {
                ...a,
                verified: true,
                verification_status: updatedItem.verification_status,
                confidence: updatedItem.confidence,
                verification_result: updatedItem.verification_result,
                evidence: updatedItem.evidence,
                verified_at: updatedItem.verified_at,
              }
            : a
        )
      );
      setExpandedId(article.id);
    } catch (err) {
      console.warn('verifyOne endpoint failed, trying text fallback:', err);
      try {
        const fbRes = await verifyAPI.verifyText({
          title: article.title,
          headline: article.title,
          text: article.summary || article.title,
          url: article.url,
        });
        setArticles((prev) =>
          prev.map((a) =>
            a.id === article.id
              ? {
                  ...a,
                  verified: true,
                  verification_status: fbRes.data.prediction,
                  confidence: fbRes.data.calibrated_confidence || fbRes.data.confidence,
                  verification_result: fbRes.data,
                  evidence: fbRes.data.evidence,
                  verified_at: new Date().toISOString(),
                }
              : a
          )
        );
        setExpandedId(article.id);
      } catch (fallbackErr) {
        console.error('Verification failed:', fallbackErr);
        setError('Verification failed for this article. Please try again.');
      }
    } finally {
      setVerifyingId(null);
    }
  };

  const toggleExpand = (id) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  return (
    <div style={{ maxWidth: '1240px', margin: '0 auto', padding: '32px 20px 60px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px', marginBottom: '28px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
            <Newspaper size={18} />
            <span>REAL-TIME MULTI-SOURCE INGESTION</span>
          </div>
          <h1 style={{ fontSize: '2.2rem', fontWeight: 800 }}>
            Live News <span style={{ color: 'var(--primary)' }}>Radar</span>
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Active RSS feeds from NYT, BBC, The Guardian, NPR, AP News, and TechCrunch, cross-referenced with PolitiFact and FactCheck.org.
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
          <span>{refreshing ? 'Syncing Feeds...' : 'Sync Live Feeds'}</span>
        </button>
      </div>

      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px 16px', background: 'var(--fake-bg)', border: '1px solid var(--fake-glow)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.88rem', marginBottom: '20px' }}>
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
              placeholder="Search live news headlines..."
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
          <p>Ingesting real-time headlines and cross-referencing registries...</p>
        </div>
      ) : articles.length === 0 ? (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
          <Newspaper size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
          <h3>No articles found</h3>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-dim)', marginBottom: '16px' }}>Sync feeds to fetch current real-world articles.</p>
          <button onClick={handleRefreshFeeds} className="btn-primary" style={{ padding: '8px 20px' }}>
            Sync Live Feeds Now
          </button>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '20px' }}>
          {articles.map((item, idx) => {
            const isExpanded = expandedId === item.id;
            const verif = item.verification_result;
            const evidenceList = item.evidence || verif?.evidence_retrieval?.evidence_list || [];
            const explanationText = verif?.explanation?.summary || verif?.explanation?.executive_summary;

            return (
              <div
                key={item.id || idx}
                className="glass-panel"
                style={{
                  padding: '20px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '14px',
                  transition: 'border-color 0.2s',
                  borderColor: isExpanded ? 'var(--primary)' : 'var(--border-glass)'
                }}
              >
                <div>
                  {/* Thumbnail Image if present */}
                  {item.image_url && (
                    <div style={{ width: '100%', height: '170px', borderRadius: '8px', overflow: 'hidden', marginBottom: '12px', background: 'rgba(0,0,0,0.2)' }}>
                      <img
                        src={item.image_url}
                        alt={item.title}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e) => { e.target.style.display = 'none'; }}
                      />
                    </div>
                  )}

                  {/* Meta row */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '6px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontSize: '0.74rem', background: 'rgba(255,255,255,0.06)', color: 'var(--primary)', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
                        {item.category || 'General'}
                      </span>
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '3px' }}>
                        <Globe size={11} />
                        {item.source || 'News Wire'}
                      </span>
                    </div>

                    {item.published_at && (
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '3px' }}>
                        <Calendar size={11} />
                        {String(item.published_at).slice(0, 16)}
                      </span>
                    )}
                  </div>

                  {/* Title */}
                  <h3 style={{ fontSize: '1.02rem', fontWeight: 600, lineHeight: 1.4, marginBottom: '8px', color: 'var(--text-main)' }}>
                    {item.title}
                  </h3>

                  {/* Summary */}
                  <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    {item.summary || item.snippet}
                  </p>
                </div>

                {/* Verification result expansion */}
                {item.verification_status && isExpanded && (
                  <div style={{ background: 'rgba(0,0,0,0.25)', borderRadius: '8px', padding: '14px', border: '1px solid var(--border-glass)', fontSize: '0.82rem' }}>
                    <div style={{ fontWeight: 600, color: 'var(--primary)', marginBottom: '6px' }}>
                      Verification Analysis:
                    </div>
                    {explanationText && (
                      <p style={{ color: 'var(--text-muted)', marginBottom: '10px', lineHeight: 1.45 }}>
                        {explanationText}
                      </p>
                    )}

                    {/* Evidence summary */}
                    {evidenceList && evidenceList.length > 0 ? (
                      <div>
                        <div style={{ fontWeight: 600, fontSize: '0.78rem', color: 'var(--text-dim)', marginBottom: '6px' }}>
                          Cross-Referenced Evidence ({evidenceList.length}):
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          {evidenceList.slice(0, 3).map((ev, eIdx) => {
                            const stanceColor = ev.stance === 'SUPPORTS' ? 'var(--real-glow)' : ev.stance === 'CONTRADICTS' ? 'var(--fake-glow)' : '#facc15';
                            return (
                              <div key={eIdx} style={{ padding: '6px 8px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', borderLeft: `3px solid ${stanceColor}` }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                                  <span style={{ fontWeight: 600, fontSize: '0.76rem', color: 'var(--text-main)' }}>
                                    {ev.publisher || 'Independent Source'}
                                  </span>
                                  <span style={{ fontSize: '0.70rem', color: stanceColor, fontWeight: 700 }}>
                                    {ev.stance}
                                  </span>
                                </div>
                                <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', margin: '2px 0' }}>
                                  {ev.title || ev.snippet}
                                </p>
                                {ev.url && (
                                  <a href={ev.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: '0.72rem', color: 'var(--primary)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                                    <span>Source Link</span>
                                    <ExternalLink size={10} />
                                  </a>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: 'var(--text-dim)', fontSize: '0.76rem', fontStyle: 'italic' }}>
                        No external fact-check contradiction found.
                      </div>
                    )}

                    {item.verified_at && (
                      <div style={{ marginTop: '10px', fontSize: '0.72rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Clock size={11} />
                        <span>Verified: {String(item.verified_at).slice(0, 19).replace('T', ' ')}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Footer status & Action */}
                <div style={{ paddingTop: '10px', borderTop: '1px solid var(--border-glass)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  {item.verification_status ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <DecisionBadge prediction={item.verification_status} size="sm" />
                      {item.confidence && (
                        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {Math.round(item.confidence * 100)}%
                        </span>
                      )}
                      <button
                        onClick={() => toggleExpand(item.id)}
                        className="btn-outline"
                        style={{ fontSize: '0.72rem', padding: '2px 8px', display: 'flex', alignItems: 'center', gap: '3px' }}
                      >
                        <span>{isExpanded ? 'Hide' : 'Evidence'}</span>
                        {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      </button>
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
                          <span>Verifying Claims...</span>
                        </>
                      ) : (
                        <>
                          <ShieldCheck size={13} color="var(--primary)" />
                          <span>Verify News</span>
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
                      <span>Read Original</span>
                      <ExternalLink size={12} />
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
