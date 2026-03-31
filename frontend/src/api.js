import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const registerUser = async (payload) => {
  const response = await api.post('/auth/register', payload);
  return response.data;
};

export const loginUser = async (payload) => {
  const response = await api.post('/auth/login', payload);
  return response.data;
};

export const logoutUser = async () => {
  const response = await api.post('/auth/logout');
  return response.data;
};

export const getCurrentUser = async () => {
  const response = await api.get('/auth/me');
  return response.data;
};

export const getRecommendations = async () => {
  const response = await api.get('/recommendations');
  return response.data;
};

export const queryNews = async (query, history = []) => {
  const started = performance.now();
  const response = await api.post('/query', { query, history });
  return {
    ...response.data,
    client_ms: Number((performance.now() - started).toFixed(1)),
  };
};

export const getChatHistory = async (limit = 50) => {
  const response = await api.get('/chats', { params: { limit } });
  return response.data;
};

export const deleteChat = async (chatId) => {
  const response = await api.delete(`/chats/${chatId}`);
  return response.data;
};

export const clearChatHistory = async () => {
  const response = await api.delete('/chats');
  return response.data;
};

export default api;
