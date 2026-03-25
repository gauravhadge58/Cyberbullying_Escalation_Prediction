import axios from "axios";

const API_BASE = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000/api";

const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

export const api = {
  getStats: () => apiClient.get("/stats").then((res) => res.data),
  getConversations: () => apiClient.get("/conversations").then((res) => res.data),
  trainModels: (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return apiClient.post("/train", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((res) => res.data);
  },
  clearData: () => apiClient.delete("/conversations").then((res) => res.data),
  clearConversation: (id) => apiClient.delete(`/conversations/${id}`).then((res) => res.data),
};
