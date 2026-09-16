import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Sparkles, Cpu } from 'lucide-react';

export default function DecisionBadge({ prediction, size = 'md', className = '' }) {
  const norm = (prediction || '').toUpperCase();

  let icon = <ShieldAlert size={size === 'lg' ? 22 : 16} />;
  let badgeClass = 'badge-fake';
  let label = 'FAKE / UNVERIFIED';

  if (norm === 'REAL' || norm === 'AUTHENTIC') {
    icon = <ShieldCheck size={size === 'lg' ? 22 : 16} />;
    badgeClass = 'badge-real';
    label = 'AUTHENTIC / REAL';
  } else if (norm === 'AI_GENERATED' || norm === 'AI-GENERATED' || norm === 'SYNTHETIC') {
    icon = <Sparkles size={size === 'lg' ? 22 : 16} />;
    badgeClass = 'badge-ai';
    label = 'AI-GENERATED / SYNTHETIC';
  } else if (norm === 'DEEPFAKE') {
    icon = <Cpu size={size === 'lg' ? 22 : 16} />;
    badgeClass = 'badge-deepfake';
    label = 'DEEPFAKE DETECTED';
  } else if (norm === 'FAKE_TAMPERED' || norm === 'FAKE_MANIPULATED' || norm === 'TAMPERED') {
    icon = <ShieldAlert size={size === 'lg' ? 22 : 16} />;
    badgeClass = 'badge-fake';
    label = 'FAKE / TAMPERED';
  } else if (norm === 'SUSPICIOUS' || norm === 'UNCERTAIN' || norm === 'DISPUTED') {
    icon = <AlertTriangle size={size === 'lg' ? 22 : 16} />;
    badgeClass = 'badge-suspicious';
    label = 'SUSPICIOUS / UNCERTAIN';
  }

  const sizeStyle = size === 'lg' 
    ? { fontSize: '1.05rem', padding: '8px 20px', gap: '10px' } 
    : size === 'sm'
    ? { fontSize: '0.75rem', padding: '2px 8px', gap: '4px' }
    : {};

  return (
    <span className={`${badgeClass} ${className}`} style={sizeStyle}>
      {icon}
      <span>{label}</span>
    </span>
  );
}
