import React from 'react';

export default function ConfidenceGauge({ confidence = 0, prediction = 'REAL', label = 'Confidence' }) {
  const pct = Math.min(Math.max(Math.round(confidence * 100), 0), 100);
  const radius = 64;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (pct / 100) * circumference;

  let strokeColor = 'var(--real-color)';
  let glowColor = 'var(--real-glow)';

  const pred = (prediction || '').toUpperCase();
  if (pred.includes('AI') || pred.includes('SYNTHETIC')) {
    strokeColor = '#c084fc';
    glowColor = 'rgba(192, 132, 252, 0.4)';
  } else if (pred.includes('DEEPFAKE')) {
    strokeColor = '#f472b6';
    glowColor = 'rgba(244, 114, 182, 0.4)';
  } else if (pred.includes('FAKE') || pred.includes('TAMPERED') || pred.includes('MANIPULATED')) {
    strokeColor = 'var(--fake-color)';
    glowColor = 'var(--fake-glow)';
  } else if (pred.includes('SUSPICIOUS') || pred.includes('UNCERTAIN')) {
    strokeColor = 'var(--suspicious-color)';
    glowColor = 'var(--suspicious-glow)';
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative' }}>
      <svg width="160" height="160" viewBox="0 0 160 160">
        {/* Background Circle */}
        <circle
          cx="80"
          cy="80"
          r={radius}
          fill="none"
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth={strokeWidth}
        />
        {/* Animated Progress Arc */}
        <circle
          cx="80"
          cy="80"
          r={radius}
          fill="none"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          transform="rotate(-90 80 80)"
          style={{
            transition: 'stroke-dashoffset 1s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.5s ease',
            filter: `drop-shadow(0 0 12px ${glowColor})`,
          }}
        />
        {/* Inner glow circle */}
        <circle
          cx="80"
          cy="80"
          r="48"
          fill="rgba(0, 0, 0, 0.4)"
          stroke="rgba(255, 255, 255, 0.05)"
          strokeWidth="1"
        />
        {/* Center Text */}
        <text
          x="80"
          y="76"
          textAnchor="middle"
          fill="#f8fafc"
          fontFamily="var(--font-display)"
          fontWeight="700"
          fontSize="28px"
        >
          {pct}%
        </text>
        <text
          x="80"
          y="98"
          textAnchor="middle"
          fill="var(--text-muted)"
          fontFamily="var(--font-body)"
          fontSize="11px"
          letterSpacing="0.05em"
        >
          CALIBRATED
        </text>
      </svg>
      <div style={{ marginTop: '8px', fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>
        {label}
      </div>
    </div>
  );
}
