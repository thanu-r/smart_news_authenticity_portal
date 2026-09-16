import React from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ConsistencyRadar({ consistencyData }) {
  if (!consistencyData) return null;

  const pairs = [
    { label: 'Headline ↔ Body', score: consistencyData.headline_body_similarity, threshold: 0.5 },
    { label: 'Text ↔ Image', score: consistencyData.text_image_consistency, threshold: 0.5 },
    { label: 'Text ↔ Video', score: consistencyData.text_video_consistency, threshold: 0.5 },
    { label: 'Claim ↔ Evidence', score: consistencyData.claim_evidence_support, threshold: 0.5 },
  ].filter(p => p.score !== undefined && p.score !== null);

  if (pairs.length === 0) return null;

  const hasConflict = consistencyData.cross_modal_conflict;

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div>
          <h4 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)' }}>Cross-Modal Semantic Alignment</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Measures contextual congruence across text, visual media, and claims</p>
        </div>
        {hasConflict ? (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--fake-color)', background: 'var(--fake-bg)', border: '1px solid var(--fake-glow)', padding: '4px 10px', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600 }}>
            <AlertCircle size={14} /> Conflict Flagged
          </span>
        ) : (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--real-color)', background: 'var(--real-bg)', border: '1px solid var(--real-glow)', padding: '4px 10px', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600 }}>
            <CheckCircle2 size={14} /> Modalities Aligned
          </span>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
        {pairs.map((pair, idx) => {
          const pct = Math.round(pair.score * 100);
          const isHealthy = pair.score >= pair.threshold;
          return (
            <div key={idx} style={{ background: 'rgba(0,0,0,0.25)', padding: '12px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: 'var(--text-muted)' }}>{pair.label}</span>
                <span style={{ fontWeight: 600, color: isHealthy ? 'var(--real-color)' : 'var(--suspicious-color)', fontFamily: 'var(--font-mono)' }}>
                  {pct}%
                </span>
              </div>
              <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                <div 
                  style={{ 
                    width: `${pct}%`, 
                    height: '100%', 
                    background: isHealthy ? 'var(--primary)' : 'var(--suspicious-color)',
                    transition: 'width 0.6s ease'
                  }} 
                />
              </div>
            </div>
          );
        })}
      </div>

      {consistencyData.conflict_reason && (
        <div style={{ marginTop: '14px', padding: '10px 14px', background: 'rgba(239, 68, 68, 0.08)', borderLeft: '3px solid var(--fake-color)', borderRadius: '4px', fontSize: '0.82rem', color: '#fca5a5' }}>
          <strong>Conflict Reason:</strong> {consistencyData.conflict_reason}
        </div>
      )}
    </div>
  );
}
