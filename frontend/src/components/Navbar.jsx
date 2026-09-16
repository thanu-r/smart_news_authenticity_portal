import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ShieldCheck, Newspaper, History, BarChart3, Settings, LogIn, LogOut, User, Menu, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const { user, isAuthenticated, isAdmin, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (path) => location.pathname === path;

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const navLinks = [
    { name: 'Verify Portal', path: '/', icon: <ShieldCheck size={18} /> },
    { name: 'Live News', path: '/news', icon: <Newspaper size={18} /> },
    { name: 'History', path: '/history', icon: <History size={18} /> },
    { name: 'Analytics', path: '/analytics', icon: <BarChart3 size={18} /> },
    ...(isAdmin ? [{ name: 'Admin', path: '/admin', icon: <Settings size={18} /> }] : []),
  ];

  return (
    <nav style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      background: 'rgba(7, 9, 14, 0.85)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderBottom: '1px solid var(--border-glass)',
      padding: '0 24px',
      height: '68px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    }}>
      {/* Brand Logo */}
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none' }}>
        <div style={{
          width: '38px',
          height: '38px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(0, 242, 254, 0.4)',
        }}>
          <ShieldCheck size={22} color="#050b14" strokeWidth={2.5} />
        </div>
        <div>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.2rem', color: '#f8fafc', letterSpacing: '-0.02em' }}>
            SMART<span style={{ color: 'var(--primary)' }}>NEWS</span>
          </span>
          <span style={{ display: 'block', fontSize: '0.65rem', color: 'var(--text-dim)', letterSpacing: '0.1em', marginTop: '-3px' }}>
            AUTHENTICITY PORTAL
          </span>
        </div>
      </Link>

      {/* Desktop Navigation Links */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }} className="desktop-nav">
        {navLinks.map((link) => {
          const active = isActive(link.path);
          return (
            <Link
              key={link.path}
              to={link.path}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.9rem',
                fontWeight: 500,
                textDecoration: 'none',
                color: active ? 'var(--primary)' : 'var(--text-muted)',
                background: active ? 'rgba(0, 242, 254, 0.08)' : 'transparent',
                border: active ? '1px solid rgba(0, 242, 254, 0.2)' : '1px solid transparent',
                transition: 'all 0.2s ease',
              }}
            >
              {link.icon}
              <span>{link.name}</span>
            </Link>
          );
        })}
      </div>

      {/* User Auth Section */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {isAuthenticated ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: 'rgba(255, 255, 255, 0.04)',
              padding: '6px 12px',
              borderRadius: '20px',
              border: '1px solid var(--border-glass)',
            }}>
              <div style={{
                width: '26px',
                height: '26px',
                borderRadius: '50%',
                background: user?.role === 'ADMIN' ? 'linear-gradient(135deg, #f59e0b, #d97706)' : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#fff',
              }}>
                {user?.username?.[0]?.toUpperCase() || 'U'}
              </div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 500 }}>
                {user?.username}
              </span>
              {user?.role === 'ADMIN' && (
                <span style={{ fontSize: '0.65rem', background: 'rgba(245, 158, 11, 0.2)', color: 'var(--suspicious-color)', padding: '2px 6px', borderRadius: '4px', fontWeight: 700 }}>
                  ADMIN
                </span>
              )}
            </div>
            <button
              onClick={handleLogout}
              className="btn-outline"
              style={{ padding: '8px 12px', fontSize: '0.82rem' }}
              title="Sign out"
            >
              <LogOut size={16} />
              <span className="desktop-only">Logout</span>
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Link to="/login" className="btn-outline" style={{ padding: '8px 16px', fontSize: '0.85rem', textDecoration: 'none' }}>
              <LogIn size={15} />
              <span>Sign In</span>
            </Link>
            <Link to="/register" className="btn-primary" style={{ padding: '8px 18px', fontSize: '0.85rem', textDecoration: 'none' }}>
              <span>Get Started</span>
            </Link>
          </div>
        )}

        {/* Mobile Toggle */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          style={{ background: 'none', border: 'none', color: 'var(--text-main)', cursor: 'pointer', display: 'none' }}
          className="mobile-toggle-btn"
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </nav>
  );
}
