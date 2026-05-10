import { useEffect, useState } from "react";
import { api } from "../api";
import ModerationPanel from "../components/ModerationPanel";

export default function Monitor() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);

  // Per-conversation moderation state: { [convId]: { data, loading, error } }
  const [moderationState, setModerationState] = useState({});

  useEffect(() => {
    fetchData();

    // Listen for WebSocket updates
    const rawWsUrl = import.meta.env.VITE_WS_URL || "/ws";
    const wsUrl = rawWsUrl.startsWith("/")
      ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}${rawWsUrl}`
      : rawWsUrl;
    let ws;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "PREDICTION_UPDATE") {
          fetchData();
        }
        // Receive real-time moderation updates from LangChain background task
        if (msg.type === "MODERATION_UPDATE" && msg.data?.conversation_id) {
          const convId = msg.data.conversation_id;
          setModerationState((prev) => ({
            ...prev,
            [convId]: { data: msg.data.moderation, loading: false, error: null },
          }));
        }
      };
    } catch (err) {
      console.error("WS Error:", err);
    }

    // Fallback polling
    const interval = setInterval(fetchData, 10000);
    return () => {
      if (ws) ws.close();
      clearInterval(interval);
    };
  }, []);

  const fetchData = async () => {
    try {
      const res = await api.getConversations();
      setData(res.conversations);

      // Hydrate moderation state from persisted DB data
      setModerationState((prev) => {
        const next = { ...prev };
        res.conversations.forEach((conv) => {
          if (conv.moderation && !next[conv.conversationId]) {
            next[conv.conversationId] = {
              data: conv.moderation,
              loading: false,
              error: null,
            };
          }
        });
        return next;
      });
    } catch (err) {
      console.error("Failed to fetch conversations", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm(`Clear all messages for conversation ${id}?`)) return;
    try {
      await api.clearConversation(id);
      fetchData();
    } catch (err) {
      console.error("Failed to delete conversation", err);
      alert("Error clearing conversation.");
    }
  };

  // Request AI analysis for a conversation on demand
  const handleRequestAnalysis = async (conv) => {
    const convId = conv.conversationId;
    setModerationState((prev) => ({
      ...prev,
      [convId]: { data: null, loading: true, error: null },
    }));
    try {
      const result = await api.getModerationExplanation(convId, {
        escalation_level: conv.escalationLevel,
        escalation_score: conv.escalationScore,
        features: conv.features || {},
        messages: (conv.messages || []).map((m) => ({
          id: m.messageId,
          conversation_id: convId,
          user_id: m.userId,
          message: m.text,
          timestamp: m.timestamp,
        })),
      });
      setModerationState((prev) => ({
        ...prev,
        [convId]: { data: result, loading: false, error: null },
      }));
    } catch (err) {
      setModerationState((prev) => ({
        ...prev,
        [convId]: {
          data: null,
          loading: false,
          error: err.message || "Analysis failed. Is GROQ_API_KEY set?",
        },
      }));
    }
  };

  const getEscalationStyle = (level) => {
    switch (level) {
      case "HIGH":   return "bg-red-100 text-red-800 border-red-200";
      case "MEDIUM": return "bg-orange-100 text-orange-800 border-orange-200";
      default:       return "bg-green-100 text-green-800 border-green-200";
    }
  };

  if (loading) return <div className="text-center mt-20 text-gray-500 animate-pulse">Loading monitor...</div>;
  if (!data?.length) return <div className="text-center mt-20 text-gray-500">No conversations monitored yet.</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Live Monitor</h2>
          <p className="text-gray-500 text-sm mt-1">
            Real-time analysis of {data.length} recent conversations.
          </p>
        </div>
        <button
          onClick={fetchData}
          className="text-sm font-medium text-primary-600 hover:text-primary-700 bg-primary-50 px-4 py-2 rounded-lg transition-colors"
        >
          Refresh Data
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {data.map((conv) => {
          const modState = moderationState[conv.conversationId] || {};
          return (
            <div
              key={conv.conversationId}
              className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden transition-all duration-200 hover:shadow-md"
            >
              {/* Conversation Header */}
              <div
                className="px-6 py-4 cursor-pointer flex justify-between items-center bg-gray-50/50 hover:bg-gray-50"
                onClick={() =>
                  setExpandedId(expandedId === conv.conversationId ? null : conv.conversationId)
                }
              >
                <div className="flex items-center gap-4">
                  <span className={`px-3 py-1 text-xs font-bold rounded-full border ${getEscalationStyle(conv.escalationLevel)}`}>
                    {conv.escalationLevel} RISK
                  </span>
                  <div>
                    <h3 className="font-semibold text-gray-900 line-clamp-1">{conv.conversationId}</h3>
                    <p className="text-xs text-gray-500 mt-1">
                      {conv.messages?.length || 0} messages • Score: {conv.escalationScore}/10 •{" "}
                      Updated: {new Date(conv.lastUpdated).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {/* AI Analysis badge if available */}
                  {modState.data && (
                    <span className="hidden sm:inline-flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded-full bg-purple-50 text-purple-700 border border-purple-200">
                      🤖 {modState.data.action_label}
                    </span>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(conv.conversationId);
                    }}
                    className="p-2 hover:bg-red-50 text-gray-400 hover:text-red-500 rounded-lg transition-colors"
                    title="Clear Room"
                  >
                    <span className="text-lg">🗑️</span>
                  </button>
                  <span className={`text-gray-400 transition-transform ${expandedId === conv.conversationId ? "rotate-180" : ""}`}>
                    ▼
                  </span>
                </div>
              </div>

              {/* Expanded Content */}
              {expandedId === conv.conversationId && (
                <div className="border-t border-gray-100">
                  {/* Messages list */}
                  <div className="px-6 py-4 bg-white max-h-72 overflow-y-auto space-y-3 animate-fade-in">
                    {conv.messages?.map((msg) => (
                      <div
                        key={msg.messageId}
                        className={`p-3 rounded-lg text-sm border-l-4 ${
                          msg.isBullying ? "bg-red-50 border-red-500" : "bg-gray-50 border-gray-300"
                        }`}
                      >
                        <div className="flex justify-between items-start mb-1">
                          <span className="font-semibold text-gray-700 text-xs">{msg.userId || "User"}</span>
                          <span className="text-gray-400 text-xs">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <p className={`text-gray-800 ${msg.isBullying ? "font-medium" : ""}`}>{msg.text}</p>
                        {msg.isBullying && (
                          <div className="mt-2 text-xs text-red-600 font-semibold flex gap-3">
                            <span>Bullying Detected</span>
                            <span className="opacity-75">Toxicity: {(msg.toxicityScore * 100).toFixed(0)}%</span>
                          </div>
                        )}
                      </div>
                    ))}
                    {!conv.messages?.length && (
                      <p className="text-sm text-gray-500 text-center py-4">No message details available.</p>
                    )}
                  </div>

                  {/* ── AI Moderation Section ── */}
                  <div className="px-6 pb-5 bg-gray-50/30 space-y-3">
                    <div className="flex items-center justify-between pt-3">
                      <p className="text-[11px] uppercase font-bold text-gray-400 tracking-widest flex items-center gap-1.5">
                        <span>🤖</span> AI Moderation Analysis
                      </p>
                      <button
                        id={`request-analysis-${conv.conversationId}`}
                        onClick={() => handleRequestAnalysis(conv)}
                        disabled={modState.loading}
                        className="text-xs font-medium px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 text-white transition-colors flex items-center gap-1.5"
                      >
                        {modState.loading ? (
                          <>
                            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                            </svg>
                            Analyzing…
                          </>
                        ) : modState.data ? "Refresh Analysis" : "Request AI Analysis"}
                      </button>
                    </div>

                    <ModerationPanel
                      moderation={modState.data || null}
                      loading={modState.loading || false}
                      error={modState.error || null}
                      riskLevel={conv.escalationLevel}
                      onRetry={() => handleRequestAnalysis(conv)}
                      compact={false}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
