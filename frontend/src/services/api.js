import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

// Attach JWT token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('smartnews_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => Promise.reject(error));

// Auth APIs
export const authAPI = {
  register: (data) => api.post('/api/auth/register', data),
  login: (data) => api.post('/api/auth/login', data),
  getMe: () => api.get('/api/auth/me'),
};

// Verification APIs
export const verifyAPI = {
  verifyText: (data) => api.post('/api/verify/text', data),
  verifyUrl: (data) => api.post('/api/verify/url', data),
  verifyImage: (formData) => api.post('/api/verify/image', formData),
  verifyVideo: (formData) => api.post('/api/verify/video', formData),
  verifyMultimodal: (formData) => api.post('/api/verify/multimodal', formData),
};

// Live News APIs
export const newsAPI = {
  getLatest: (limit = 20, category = '') => 
    api.get(`/api/news/latest?limit=${limit}${category ? `&category=${category}` : ''}`),
  search: (query, limit = 20) => 
    api.get(`/api/news/search?q=${encodeURIComponent(query)}&limit=${limit}`),
  refresh: () => api.post('/api/news/refresh'),
  verifyOne: (id) => api.post(`/api/news/verify-one/${id}`),
};

// History APIs
export const historyAPI = {
  getHistory: (limit = 50, skip = 0, modality = '') => 
    api.get(`/api/history?limit=${limit}&skip=${skip}${modality ? `&modality=${modality}` : ''}`),
  getById: (id) => api.get(`/api/history/${id}`),
  submitFeedback: (id, feedback) => api.post(`/api/history/${id}/feedback`, feedback),
};

// Evidence APIs
export const evidenceAPI = {
  query: (claim, maxResults = 5) => 
    api.post('/api/evidence/query', { claim, max_results: maxResults }),
  extractClaims: (text) => 
    api.post('/api/evidence/extract-claims', { text }),
};

// Model Metrics APIs
export const modelsAPI = {
  getMetrics: () => api.get('/api/models/metrics'),
  getStatus: () => api.get('/api/models/status'),
};

// Admin APIs
export const adminAPI = {
  getStats: () => api.get('/api/admin/metrics'),
  getUsers: () => api.get('/api/admin/users'),
  getFlagged: (status = '') => api.get(`/api/admin/flagged-records${status ? `?status=${status}` : ''}`),
  triggerRetrain: (modality) => api.post('/api/admin/retrain', { modality }),
};

export default api;
