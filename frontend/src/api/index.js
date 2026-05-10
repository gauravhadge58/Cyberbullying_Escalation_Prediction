import axios from "axios";

const API_BASE = import.meta.env.VITE_BACKEND_URL || "/api";

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

  // Model Registry
  getTrainingProgress: () => apiClient.get("/train/progress").then((res) => res.data),
  listModels: () => apiClient.get("/models").then((res) => res.data),
  activateModel: (id) => apiClient.post(`/models/${id}/activate`).then((res) => res.data),
  deleteModel: (id) => apiClient.delete(`/models/${id}`).then((res) => res.data),

  // Gemini Chatbot
  chat: (message, history = []) =>
    apiClient.post("/chat", { message, history }).then((res) => res.data),

  // ── LangChain Moderation Layer ────────────────────────────────────────────
  /** Check if the LangChain LLM provider is reachable */
  getModerationHealth: () =>
    apiClient.get("/moderation/health").then((res) => res.data),

  /**
   * Request an on-demand AI moderation explanation for a conversation.
   * @param {string} convId - conversation ID
   * @param {object} payload - { escalation_level, features, messages }
   */
  getModerationExplanation: (convId, payload) =>
    apiClient.post("/moderation/explain", {
      conversation_id: convId,
      ...payload,
    }, { timeout: 35000 }).then((res) => res.data),

  /**
   * Get a short AI-generated summary of a conversation's escalation pattern.
   * @param {string} convId - conversation ID
   */
  getConversationSummary: (convId) =>
    apiClient.get(`/moderation/summary/${convId}`, { timeout: 35000 }).then((res) => res.data),
};

