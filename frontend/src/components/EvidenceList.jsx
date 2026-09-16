import React from 'react';
import { ExternalLink, Check, X, Minus, BookOpen } from 'lucide-react';

export default function EvidenceList({ evidence = [] }) {
  if (!evidence || evidence.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '20px', textAlign: 'center', color: 'var(--text-dim)' }}>
        <BookOpen size={28} style={{ margin: '0 auto 8px', opacity: 0.5 }} />
        <p style={{ fontSize: '0.9rem' }}>No external evidence documents matched for this claim.</p>
      </div>
    );
  }

  const getStanceBadge = (stance) => {
    const s = (stance || 'NEUTRAL').toUpperCase();
    if (s.includes('SUPPORT')) {
      return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', background: 'var(--real-bg)', color: 'var(--real-color)', border: '1px solid var(--real-glow)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>
          <Check size={12} /> SUPPORTS
        </span>
      );
    }
    if (s.includes('REFUTE') || s.includes('CONTRADICT')) {
      return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', background: 'var(--fake-bg)', color: 'var(--fake-color)', border: '1px solid var(--fake-glow)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>
          <X size={12} /> REFUTES
        </span>
      );
    }
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', background: 'rgba(255,255,255,0.06)', color: 'var(--text-muted)', border: '1px solid var(--border-glass)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>
        <Minus size={12} /> NEUTRAL / MENTIONS
      </span>
    );
  };

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BookOpen size={18} color="var(--primary)" />
          <h4 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)' }}>Retrieved Ground-Truth Evidence</h4>
        </div>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{evidence.length} sources matched</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {evidence.map((item, idx) => (
          <div 
            key={idx} 
            style={{ 
              background: 'rgba(0,0,0,0.2)', 
              padding: '12px 16px', 
              borderRadius: '8px', 
              border: '1px solid rgba(255,255,255,0.05)',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
              <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-main)' }}>
                {item.source || item.outlet || 'Verified Registry'}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {getStanceBadge(item.stance)}
                {item.similarity !== undefined && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                    Sim: {Math.round(item.similarity * 100)}%
                  </span>
                )}
              </div>
            </div>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              "{item.snippet || item.title || item.claim}"
            </p>

            {item.url && (
              <a 
                href={item.url} 
                target="_blank" 
                rel="noopener noreferrer" 
                style={{ 
                  display: 'inline-flex', 
                  alignItems: 'center', 
                  gap: '4px', 
                  fontSize: '0.78rem', 
                  color: 'var(--primary)', 
                  textDecoration: 'none',
                  marginTop: '2px'
                }}
              >
                <span>View Source Article</span>
                <ExternalLink size={12} />
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
