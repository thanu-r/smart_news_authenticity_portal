import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ShieldCheck, User, Mail, Lock, ArrowRight, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Register() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await register(username.trim(), email.trim(), password);
      setSuccess(true);
      setTimeout(() => {
        navigate('/login');
      }, 1500);
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.response?.data?.detail || 'Failed to create account. Please check your details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: 'calc(100vh - 160px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
      <div className="glass-panel glow-panel" style={{ maxWidth: '440px', width: '100%', padding: '36px' }}>
        <div style={{ textAlign: 'center', marginBottom: '28px' }}>
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '14px',
            boxShadow: '0 0 20px rgba(0, 242, 254, 0.4)',
          }}>
            <ShieldCheck size={28} color="#050b14" strokeWidth={2.5} />
          </div>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-main)' }}>
            Create Account
          </h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Join SmartNews research portal to audit news authenticity.
          </p>
        </div>

        {error && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: 'var(--fake-bg)', border: '1px solid var(--fake-glow)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.85rem', marginBottom: '20px' }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: 'var(--real-bg)', border: '1px solid var(--real-glow)', borderRadius: '8px', color: 'var(--real-color)', fontSize: '0.85rem', marginBottom: '20px' }}>
            <CheckCircle2 size={16} />
            <span>Account created successfully! Redirecting to sign in...</span>
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>
              Username
            </label>
            <div style={{ position: 'relative' }}>
              <User size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
              <input
                type="text"
                required
                className="form-input"
                style={{ paddingLeft: '38px' }}
                placeholder="researcher_user"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>
              Email Address
            </label>
            <div style={{ position: 'relative' }}>
              <Mail size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
              <input
                type="email"
                required
                className="form-input"
                style={{ paddingLeft: '38px' }}
                placeholder="researcher@university.edu"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
              <input
                type="password"
                required
                className="form-input"
                style={{ paddingLeft: '38px' }}
                placeholder="Minimum 8 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn-primary"
            style={{ width: '100%', padding: '12px', marginTop: '6px' }}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                <span>Creating Account...</span>
              </>
            ) : (
              <>
                <span>Register</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: '22px', fontSize: '0.85rem', color: 'var(--text-dim)' }}>
          Already have an account?{' '}
          <Link to="/login" style={{ color: 'var(--primary)', fontWeight: 600, textDecoration: 'none' }}>
            Sign in here
          </Link>
        </div>
      </div>
    </div>
  );
}
