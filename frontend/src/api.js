import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const queryNews = async (query) => {
  const response = await api.post('/query', { query });
  return response.data;
};

export const ingestArticles = async (articles, force = false) => {
  const response = await api.post('/ingest', { articles, force });
  return response.data;
};

export const runDemo = async () => {
  const response = await api.get('/demo');
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
