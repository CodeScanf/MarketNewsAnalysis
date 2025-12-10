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
