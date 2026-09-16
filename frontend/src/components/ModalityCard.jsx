import React from 'react';
import { FileText, Image as ImageIcon, Video, Globe, Activity } from 'lucide-react';
import DecisionBadge from './DecisionBadge';

export default function ModalityCard({ title, modality, data, weight }) {
  if (!data) return null;

  const getIcon = () => {
    switch (modality) {
      case 'text': return <FileText size={20} color="var(--primary)" />;
      case 'image': return <ImageIcon size={20} color="#ec4899" />;
      case 'video': return <Video size={20} color="#a855f7" />;
      case 'source': return <Globe size={20} color="#3b82f6" />;
      default: return <Activity size={20} color="var(--primary)" />;
    }
  };

  const prediction = data.prediction || (data.credible ? 'REAL' : 'SUSPICIOUS');
  const confidence = data.confidence !== undefined ? Math.round(data.confidence * 100) : null;
  const rawProb = data.raw_probability !== undefined ? Math.round(data.raw_probability * 100) : null;

  return (
    <div className="glass-panel" style={{ padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {getIcon()}
          <h4 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)' }}>{title}</h4>
          {weight !== undefined && (
            <span style={{ fontSize: '0.75rem', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: '4px', color: 'var(--text-muted)' }}>
              Weight: {Math.round(weight * 100)}%
            </span>
          )}
        </div>
        <DecisionBadge prediction={prediction} size="sm" />
      </div>

      {confidence !== null && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
            <span>Calibrated Confidence</span>
            <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{confidence}%</span>
          </div>
          <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
            <div 
              style={{ 
                width: `${confidence}%`, 
                height: '100%', 
                background: prediction === 'REAL' ? 'var(--real-color)' : prediction === 'FAKE' ? 'var(--fake-color)' : 'var(--suspicious-color)',
                transition: 'width 0.8s ease'
              }} 
            />
          </div>
        </div>
      )}

      {/* Specific Modality Signals */}
      <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '4px', background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px' }}>
        {modality === 'text' && (
          <>
            {data.model && <div>Model: <span style={{ color: 'var(--text-main)' }}>{data.model}</span></div>}
            {data.chunks_evaluated !== undefined && <div>Chunks Evaluated: <span style={{ color: 'var(--text-main)' }}>{data.chunks_evaluated}</span></div>}
            {data.top_keywords && data.top_keywords.length > 0 && (
              <div>Key Tokens: <span style={{ color: 'var(--primary)' }}>{data.top_keywords.slice(0, 4).join(', ')}</span></div>
            )}
          </>
        )}

        {modality === 'image' && (
          <>
            {(data.ela_mean !== undefined || data.forensic_signals?.ela_error_score !== undefined) && (
              <div>ELA Residual Score: <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>
                {(data.ela_mean ?? data.forensic_signals?.ela_error_score)?.toFixed(3)}
              </span></div>
            )}
            {(data.noise_variance !== undefined || data.forensic_signals?.noise_inconsistency !== undefined) && (
              <div>Laplacian Variance: <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>
                {(data.noise_variance ?? data.forensic_signals?.noise_inconsistency)?.toFixed(3)}
              </span></div>
            )}
            {data.forensic_signals?.sharpness_index !== undefined && (
              <div>Sharpness Index: <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>
                {data.forensic_signals.sharpness_index.toFixed(2)}
              </span></div>
            )}
            {data.forensic_signals?.metadata?.editing_software_detected !== undefined && (
              <div>Editing Software: <span style={{ color: data.forensic_signals.metadata.editing_software_detected ? 'var(--fake-color)' : 'var(--real-color)', fontWeight: 600 }}>
                {data.forensic_signals.metadata.editing_software_detected ? `Detected (${data.forensic_signals.metadata.software_name || 'Traces'})` : 'None (Camera Native)'}
              </span></div>
            )}
            {data.forensic_signals?.anomalies_detected && data.forensic_signals.anomalies_detected.length > 0 && (
              <div style={{ color: 'var(--fake-color)', fontSize: '0.78rem', marginTop: '2px' }}>
                ⚠️ {data.forensic_signals.anomalies_detected[0]}
              </div>
            )}
          </>
        )}

        {modality === 'video' && (
          <>
            {data.frames_analyzed !== undefined && <div>Frames Analyzed: <span style={{ color: 'var(--text-main)' }}>{data.frames_analyzed}</span></div>}
            {data.temporal_flicker !== undefined && <div>Temporal Flicker: <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{data.temporal_flicker.toFixed(3)}</span></div>}
            {data.deepfake_risk && <div>Risk Level: <span style={{ color: 'var(--fake-color)', fontWeight: 600 }}>{data.deepfake_risk}</span></div>}
          </>
        )}

        {modality === 'source' && (
          <>
            {data.domain && <div>Domain: <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{data.domain}</span></div>}
            {data.reputation_tier && <div>Reputation Tier: <span style={{ color: 'var(--primary)' }}>{data.reputation_tier}</span></div>}
            {data.satire_detected !== undefined && <div>Satire Flag: <span style={{ color: data.satire_detected ? 'var(--fake-color)' : 'var(--real-color)' }}>{data.satire_detected ? 'Satirical Source' : 'Standard'}</span></div>}
          </>
        )}
      </div>
    </div>
  );
}
