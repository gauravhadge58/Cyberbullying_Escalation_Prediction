import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { api } from "../api";
import ModerationPanel from "../components/ModerationPanel";

const API_URL = import.meta.env.VITE_BACKEND_URL || "/api";
const rawWsUrl = import.meta.env.VITE_WS_URL || "/ws";
const WS_URL = rawWsUrl.startsWith("/") 
  ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}${rawWsUrl}`
  : rawWsUrl;

export default function Simulator() {
  const [roomId, setRoomId] = useState("");
  const [messages, setMessages] = useState([]);
  const [escalationLevel, setEscalationLevel] = useState("LOW");
  const [inputText, setInputText] = useState("");
  const [username, setUsername] = useState("");
  const [isJoined, setIsJoined] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [muteRemaining, setMuteRemaining] = useState(0);

  // ── LangChain Moderation State ──────────────────────────────────────────
  const [moderation, setModeration] = useState(null);
  const [moderationLoading, setModerationLoading] = useState(false);
  const [moderationError, setModerationError] = useState(null);
  const [moderationOpen, setModerationOpen] = useState(true);

  const scrollRef = useRef(null);
  const currentRoomId = useRef(roomId);
  useEffect(() => { currentRoomId.current = roomId; }, [roomId]);

  // Fetch initial messages for the demo room
  useEffect(() => {
    if (isJoined) {
      // Simulator always starts fresh — don't load old MongoDB messages.
      setMessages([]);
      setEscalationLevel("LOW");
      setModeration(null);
      setModerationError(null);

      // WebSocket connection
      const ws = new WebSocket(WS_URL);
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);

        // ── PREDICTION_UPDATE: new ML results ─────────────────────────────
        if (payload.type === "PREDICTION_UPDATE") {
          const newDemoMsgs = payload.data.messages.filter(
            (m) => m.conversation_id === currentRoomId.current
          );

          if (newDemoMsgs.length > 0) {
            setMessages((prev) => {
              const prevIds = new Set(prev.map((m) => m.messageId));
              const additions = newDemoMsgs
                .filter((m) => !prevIds.has(m.id))
                .map((m) => ({
                  messageId: m.id,
                  conversationId: m.conversation_id,
                  userId: m.user_id,
                  text: m.message,
                  timestamp: m.timestamp,
                  isBullying: m.is_bullying,
                  toxicityScore: m.toxicity_score,
                  confidence: m.confidence,
                }));
              return [...prev, ...additions].sort(
                (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
              );
            });
          }

          const demoConvUpdate = payload.data.conversations.find(
            (c) => c.conversation_id === currentRoomId.current
          );
          if (demoConvUpdate) {
            setEscalationLevel(demoConvUpdate.escalation_level);
            // Show loading indicator while LangChain processes in background
            setModerationLoading(true);
            setModerationError(null);
          }
        }

        // ── MODERATION_UPDATE: LangChain explanation ready ─────────────────
        if (
          payload.type === "MODERATION_UPDATE" &&
          payload.data?.conversation_id === currentRoomId.current
        ) {
          setModeration(payload.data.moderation);
          setModerationLoading(false);
          setModerationError(null);
        }
      };

      return () => ws.close();
    }
  }, [isJoined]);

  // Handle mute countdown
  useEffect(() => {
    let interval;
    if (isMuted && muteRemaining > 0) {
      interval = setInterval(() => {
        setMuteRemaining((prev) => {
          if (prev <= 1) { setIsMuted(false); return 0; }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isMuted, muteRemaining]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  const handleJoin = (e) => {
    e.preventDefault();
    if (username.trim() && roomId.trim()) setIsJoined(true);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || loading) return;

    const newMsg = {
      id: `msg_${Math.random().toString(36).substr(2, 9)}`,
      conversation_id: roomId,
      user_id: username,
      message: inputText.trim(),
      timestamp: new Date().toISOString(),
    };

    const optimisticMsg = {
      messageId: newMsg.id,
      conversationId: newMsg.conversation_id,
      userId: newMsg.user_id,
      text: newMsg.message,
      timestamp: newMsg.timestamp,
      isBullying: false,
      isOptimistic: true,
    };
    setMessages((prev) => [...prev, optimisticMsg]);
    setInputText("");

    setLoading(true);
    try {
      const contextMessages = [
        ...messages.map((m) => ({
          id: m.messageId,
          conversation_id: m.conversationId,
          user_id: m.userId,
          message: m.text,
          timestamp: m.timestamp,
        })),
        newMsg,
      ];

      const res = await axios.post(`${API_URL}/predict`, {
        messages: contextMessages,
        new_message_id: newMsg.id,
      });

      // Update with server results (toxicity, etc)
      if (res.data.messages?.length > 0) {
        const result = res.data.messages[0];
        setMessages((prev) =>
          prev.map((m) =>
            m.messageId === newMsg.id
              ? { ...m, isBullying: result.is_bullying, toxicityScore: result.toxicity_score, isOptimistic: false }
              : m
          )
        );
      }

      // Update escalation level from HTTP response
      if (res.data.conversations?.length > 0) {
        const demoConvUpdate = res.data.conversations.find((c) => c.conversation_id === roomId);
        if (demoConvUpdate) {
          setEscalationLevel(demoConvUpdate.escalation_level);
          // Show loading while waiting for MODERATION_UPDATE via WS
          setModerationLoading(true);
          setModerationError(null);
        }
      }
    } catch (err) {
      setMessages((prev) => prev.filter((m) => m.messageId !== newMsg.id));
      if (err.response?.status === 403 && err.response?.data?.muted) {
        setIsMuted(true);
        setMuteRemaining(err.response.data.remaining || 600);
        alert(err.response.data.error);
      } else {
        console.error("Failed to send message", err);
        alert(`Error sending message: ${err.message}. Please check if the backend is reachable at ${API_URL}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm("Clear all messages in this room?")) return;
    try {
      setLoading(true);
      await api.clearConversation(roomId);
      setMessages([]);
      setEscalationLevel("LOW");
      setModeration(null);
      setModerationError(null);
    } catch (err) {
      console.error("Failed to clear chat", err);
      alert("Error clearing chat.");
    } finally {
      setLoading(false);
    }
  };

  // Manual re-request AI analysis
  const handleRetryModeration = async () => {
    if (!messages.length) return;
    setModerationLoading(true);
    setModerationError(null);
    try {
      const result = await api.getModerationExplanation(roomId, {
        escalation_level: escalationLevel,
        features: {},
        messages: messages.map((m) => ({
          id: m.messageId,
          conversation_id: m.conversationId,
          user_id: m.userId,
          message: m.text,
          timestamp: m.timestamp,
        })),
      });
      setModeration(result);
    } catch (err) {
      setModerationError(err.message || "Analysis failed. Is GROQ_API_KEY set?");
    } finally {
      setModerationLoading(false);
    }
  };

  // ── Join Screen ────────────────────────────────────────────────────────────
  if (!isJoined) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-200 w-full max-w-md text-center animate-fade-in">
          <div className="text-4xl mb-4">💬</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Join Simulator</h2>
          <p className="text-gray-500 mb-6">Enter a room ID and username to join the live model test room.</p>
          <form onSubmit={handleJoin} className="space-y-4">
            <input
              type="text"
              placeholder="Room ID (e.g. live_demo_room)"
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500"
              value={roomId}
              onChange={(e) => setRoomId(e.target.value)}
              required
            />
            <input
              type="text"
              placeholder="Username"
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <button
              type="submit"
              className="w-full bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 rounded-lg transition-colors"
            >
              Enter Chat Room
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Determine header color based on escalation
  const headerColors = {
    LOW: "bg-success text-white",
    MEDIUM: "bg-orange-500 text-white border-b-4 border-orange-600",
    HIGH: "bg-danger text-white border-b-4 border-red-700",
  };

  return (
    <div className="h-full flex gap-4">
      {/* ── Left: Chat Panel ── */}
      <div className="flex-1 flex flex-col bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-fade-in min-w-0">
        {/* Header */}
        <div className={`px-4 md:px-6 py-3 md:py-4 flex justify-between items-center transition-colors duration-500 ${headerColors[escalationLevel]}`}>
          <div>
            <h2 className="text-base md:text-lg font-bold truncate max-w-[150px] md:max-w-none">
              Test Room: {roomId}
            </h2>
            <p className="text-xs opacity-90">User: {username}</p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <button
              onClick={handleClear}
              disabled={loading || messages.length === 0}
              className="text-[10px] font-bold uppercase bg-white/20 hover:bg-white/30 px-2 py-1 rounded transition-colors disabled:opacity-50"
            >
              Clear Chat
            </button>
            <div className="text-right">
              <span className="text-xs font-semibold uppercase opacity-80 block mb-0.5">Escalation Risk</span>
              <span className="font-bold text-xl drop-shadow-sm">{escalationLevel}</span>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <span className="text-4xl mb-2">✨</span>
              <p>No messages yet. Start the conversation!</p>
            </div>
          ) : (
            messages.map((msg) => {
              const isMe = msg.userId === username;
              return (
                <div key={msg.messageId} className={`flex ${isMe ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[70%] rounded-2xl px-5 py-3 shadow-sm ${
                      msg.isBullying
                        ? "bg-red-50 border border-red-200"
                        : isMe
                        ? "bg-primary-500 text-white"
                        : "bg-white border border-gray-200"
                    } ${msg.isOptimistic ? "opacity-70" : "opacity-100"}`}
                  >
                    {!isMe && (
                      <div className="text-xs font-semibold text-gray-500 mb-1">{msg.userId}</div>
                    )}
                    <p className={msg.isBullying ? "text-red-700 font-medium" : ""}>{msg.text}</p>
                    {msg.isBullying && (
                      <div className="mt-2 text-[10px] uppercase font-bold text-red-500 tracking-wider">
                        ⚠️ Toxic Content ({(msg.toxicityScore * 100).toFixed(0)}%)
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Input Area */}
        <form onSubmit={handleSend} className="p-4 bg-white border-t border-gray-200 flex gap-3">
          <input
            type="text"
            placeholder={isMuted ? `Muted: Wait ${muteRemaining}s` : "Type a message to test the prediction model..."}
            className={`flex-1 px-4 py-3 rounded-xl border focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors ${
              isMuted ? "bg-red-50 border-red-200 cursor-not-allowed" : "bg-gray-50 border-gray-300 focus:bg-white"
            }`}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={loading || isMuted}
            autoFocus
          />
          <button
            type="submit"
            disabled={!inputText.trim() || loading || isMuted}
            className="px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-all shadow-sm active:scale-95 flex items-center gap-2"
          >
            {loading ? "Sending..." : isMuted ? "Muted" : "Send"}
            <span className="text-xl">{isMuted ? "🔇" : "✈️"}</span>
          </button>
        </form>
      </div>

      {/* ── Right: AI Moderation Panel ── */}
      <div className="w-80 xl:w-96 shrink-0 flex flex-col gap-3">
        {/* Panel toggle header */}
        <div
          className="flex items-center justify-between bg-white rounded-xl border border-gray-200 px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors shadow-sm"
          onClick={() => setModerationOpen((o) => !o)}
          id="moderation-panel-toggle"
        >
          <div className="flex items-center gap-2">
            <span className="text-lg">🛡️</span>
            <span className="font-semibold text-gray-800 text-sm">AI Moderator</span>
            {moderationLoading && (
              <svg className="w-3.5 h-3.5 animate-spin text-purple-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
            )}
          </div>
          <span className={`text-gray-400 text-xs transition-transform ${moderationOpen ? "rotate-180" : ""}`}>▼</span>
        </div>

        {/* ModerationPanel */}
        {moderationOpen && (
          <div className="flex-1 overflow-y-auto space-y-3">
            {/* Show panel when loading OR has data OR has error */}
            {(moderationLoading || moderation || moderationError) ? (
              <ModerationPanel
                moderation={moderation}
                loading={moderationLoading}
                error={moderationError}
                riskLevel={escalationLevel}
                onRetry={messages.length > 0 ? handleRetryModeration : null}
              />
            ) : (
              <div className="bg-white rounded-2xl border border-gray-100 p-5 text-center shadow-sm">
                <div className="text-3xl mb-2">🤖</div>
                <p className="text-sm text-gray-500 font-medium">AI Analysis Ready</p>
                <p className="text-xs text-gray-400 mt-1">
                  Send messages to trigger the full ML + LangChain pipeline.
                </p>
                <p className="text-xs text-gray-400 mt-3 italic">
                  BERT → BiLSTM → Random Forest → LangChain
                </p>
              </div>
            )}

            {/* ML Signal Summary */}
            {messages.length > 0 && !moderationLoading && (
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-2">
                <p className="text-[10px] uppercase font-bold text-gray-400 tracking-widest">ML Signals</p>
                <div className="grid grid-cols-2 gap-2">
                  <div className="text-center">
                    <p className="text-lg font-bold text-gray-800">{messages.length}</p>
                    <p className="text-xs text-gray-400">Messages</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold text-red-600">
                      {messages.filter((m) => m.isBullying).length}
                    </p>
                    <p className="text-xs text-gray-400">Flagged</p>
                  </div>
                  <div className="col-span-2 text-center">
                    <span
                      className={`text-xs font-bold px-3 py-1 rounded-full ${
                        escalationLevel === "HIGH"
                          ? "bg-red-100 text-red-700"
                          : escalationLevel === "MEDIUM"
                          ? "bg-orange-100 text-orange-700"
                          : "bg-emerald-100 text-emerald-700"
                      }`}
                    >
                      {escalationLevel} RISK
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
