import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { api } from "../api";

const API_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000/api";
const rawWsUrl = import.meta.env.VITE_WS_URL || "ws://localhost:5000";
const WS_URL = rawWsUrl.startsWith("/") 
  ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}${rawWsUrl}`
  : rawWsUrl;
const CONVERSATION_ID = "live_demo_room";

export default function Simulator() {
  const [messages, setMessages] = useState([]);
  const [escalationLevel, setEscalationLevel] = useState("LOW");
  const [inputText, setInputText] = useState("");
  const [username, setUsername] = useState("");
  const [isJoined, setIsJoined] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [muteRemaining, setMuteRemaining] = useState(0);
  const scrollRef = useRef(null);

  // Fetch initial messages for the demo room
  useEffect(() => {
    if (isJoined) {
      axios.get(`${API_URL}/conversations`).then((res) => {
        const demoConv = res.data.conversations.find((c) => c.conversationId === CONVERSATION_ID);
        if (demoConv) {
          setMessages(demoConv.messages || []);
          setEscalationLevel(demoConv.escalationLevel || "LOW");
        }
      }).catch(console.error);

      // WebSocket connection
      const ws = new WebSocket(WS_URL);
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "PREDICTION_UPDATE") {
          // Check if any of the new messages belong to our demo room
          const newDemoMsgs = payload.data.messages.filter((m) => m.conversation_id === CONVERSATION_ID);
          
          if (newDemoMsgs.length > 0) {
            // Update messages (avoid duplicates using messageId)
            setMessages((prev) => {
              const prevIds = new Set(prev.map(m => m.messageId));
              const additions = newDemoMsgs.filter(m => !prevIds.has(m.id)).map(m => ({
                messageId: m.id,
                conversationId: m.conversation_id,
                userId: m.user_id,
                text: m.message,
                timestamp: m.timestamp,
                isBullying: m.is_bullying,
                toxicityScore: m.toxicity_score,
                confidence: m.confidence
              }));
              return [...prev, ...additions].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            });
          }

          // Check for escalation update
          const demoConvUpdate = payload.data.conversations.find((c) => c.conversation_id === CONVERSATION_ID);
          if (demoConvUpdate) {
            setEscalationLevel(demoConvUpdate.escalation_level);
          }
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
          if (prev <= 1) {
            setIsMuted(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isMuted, muteRemaining]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleJoin = (e) => {
    e.preventDefault();
    if (username.trim()) setIsJoined(true);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || loading) return;

    const newMsg = {
      id: `msg_${crypto.randomUUID()}`,
      conversation_id: CONVERSATION_ID,
      user_id: username,
      message: inputText.trim(),
      timestamp: new Date().toISOString()
    };

    setLoading(true);
    try {
      // Send to prediction API
      await axios.post(`${API_URL}/predict`, {
        messages: [newMsg]
      });
      setInputText("");
    } catch (err) {
      if (err.response?.status === 403 && err.response?.data?.muted) {
        setIsMuted(true);
        setMuteRemaining(err.response.data.remaining || 600);
        alert(err.response.data.error);
      } else {
        console.error("Failed to send message", err);
        alert("Error sending message. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm("Clear all messages in this room?")) return;
    try {
      setLoading(true);
      await api.clearConversation(CONVERSATION_ID);
      setMessages([]);
      setEscalationLevel("LOW");
    } catch (err) {
      console.error("Failed to clear chat", err);
      alert("Error clearing chat.");
    } finally {
      setLoading(false);
    }
  };

  if (!isJoined) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-200 w-full max-w-md text-center animate-fade-in">
          <div className="text-4xl mb-4">💬</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Join Simulator</h2>
          <p className="text-gray-500 mb-6">Enter a username to join the live model test room.</p>
          <form onSubmit={handleJoin} className="space-y-4">
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
    MEDIUM: "bg-warning text-white",
    HIGH: "bg-danger text-white border-b-4 border-red-700"
  };

  return (
    <div className="h-full flex flex-col bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-fade-in">
      {/* Header */}
      <div className={`px-4 md:px-6 py-3 md:py-4 flex justify-between items-center transition-colors duration-500 ${headerColors[escalationLevel]}`}>
        <div>
          <h2 className="text-base md:text-lg font-bold truncate max-w-[150px] md:max-w-none">Test Room: {CONVERSATION_ID}</h2>
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
                      : (isMe ? "bg-primary-500 text-white" : "bg-white border border-gray-200")
                  }`}
                >
                  {!isMe && (
                    <div className="text-xs font-semibold text-gray-500 mb-1">
                      {msg.userId}
                    </div>
                  )}
                  <p className={msg.isBullying ? "text-red-700 font-medium" : ""}>
                    {msg.text}
                  </p>
                  
                  {/* ML Confidence Details */}
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
          {loading ? "Sending..." : (isMuted ? "Muted" : "Send")} 
          <span className="text-xl">{isMuted ? "🔇" : "✈️"}</span>
        </button>
      </form>
    </div>
  );
}
