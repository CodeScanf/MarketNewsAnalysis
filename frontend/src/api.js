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

export default api;
