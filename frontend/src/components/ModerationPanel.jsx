/**
 * ModerationPanel — AI-powered moderation explanation card.
 *
 * Displays:
 *  - Risk level badge with animated pulse
 *  - AI-generated explanation (with typewriter reveal animation)
 *  - Escalation reasoning bullet list
 *  - Suggested moderation action (colour-coded)
 *  - Dashboard headline summary
 *  - Provider badge
 *
 * States:
 *  - loading: skeleton UI while LangChain generates the explanation
 *  - result:  populated explanation
 *  - error:   fallback message with retry option
 *  - null:    hidden (no moderation data yet)
 */
import { useState, useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Action config
// ─────────────────────────────────────────────────────────────────────────────
const ACTION_CONFIG = {
  monitor: {
    label: "👁️ Monitor",
    bg: "bg-blue-50",
    border: "border-blue-300",
    text: "text-blue-800",
    badge: "bg-blue-100 text-blue-700",
    description: "Keep watching — no immediate action required.",
  },
  warn_user: {
    label: "⚠️ Warn User",
    bg: "bg-yellow-50",
    border: "border-yellow-300",
    text: "text-yellow-800",
    badge: "bg-yellow-100 text-yellow-700",
    description: "Send a formal warning to the offending user.",
  },
  temporary_mute: {
    label: "🔇 Temporary Mute",
    bg: "bg-orange-50",
    border: "border-orange-300",
    text: "text-orange-900",
    badge: "bg-orange-100 text-orange-800",
    description: "Mute the user temporarily (10 minutes).",
  },
  escalate_to_human: {
    label: "🚨 Escalate to Human",
    bg: "bg-red-50",
    border: "border-red-300",
    text: "text-red-900",
    badge: "bg-red-100 text-red-800",
    description: "Immediate human moderator intervention required.",
  },
};

const RISK_COLORS = {
  HIGH: { dot: "bg-red-500", ring: "ring-red-500/30", text: "text-red-700" },
  MEDIUM: { dot: "bg-orange-400", ring: "ring-orange-400/30", text: "text-orange-700" },
  LOW: { dot: "bg-emerald-500", ring: "ring-emerald-500/30", text: "text-emerald-700" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Typewriter hook — reveals text character by character
// ─────────────────────────────────────────────────────────────────────────────
function useTypewriter(text, speed = 18) {
  const [displayed, setDisplayed] = useState("");
  const indexRef = useRef(0);
  const textRef = useRef(text);

  useEffect(() => {
    if (!text) { setDisplayed(""); return; }
    if (text !== textRef.current) {
      textRef.current = text;
      indexRef.current = 0;
      setDisplayed("");
    }
    const interval = setInterval(() => {
      if (indexRef.current < textRef.current.length) {
        setDisplayed(textRef.current.slice(0, indexRef.current + 1));
        indexRef.current += 1;
      } else {
        clearInterval(interval);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  return displayed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Skeleton loader
// ─────────────────────────────────────────────────────────────────────────────
function Skeleton({ className = "" }) {
  return (
    <div
      className={`animate-pulse rounded bg-gradient-to-r from-gray-100 via-gray-200 to-gray-100 bg-[length:400%_100%] ${className}`}
      style={{ animation: "shimmer 1.5s infinite", backgroundSize: "400% 100%" }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function ModerationPanel({
  moderation,        // null | { explanation, reasoning, suggested_action, action_label, summary, provider }
  loading = false,   // true while LangChain is generating
  error = null,      // error string
  riskLevel = "LOW", // from ML pipeline
  onRetry = null,    // callback to re-request explanation
  compact = false,   // compact mode for Monitor page
}) {
  const action = ACTION_CONFIG[moderation?.suggested_action] || ACTION_CONFIG.monitor;
  const riskColor = RISK_COLORS[riskLevel] || RISK_COLORS.LOW;
  const revealedExplanation = useTypewriter(moderation?.explanation || "", 14);

  // Don't render anything if no moderation data and not loading
  if (!moderation && !loading && !error) return null;

  // Parse reasoning bullets
  const reasoningBullets = moderation?.reasoning
    ? moderation.reasoning.split("\n").filter((l) => l.trim())
    : [];

  return (
    <div
      id="moderation-panel"
      className={`rounded-2xl border-2 shadow-sm transition-all duration-500 ${
        moderation
          ? `${action.bg} ${action.border}`
          : "bg-gray-50 border-gray-200"
      } ${compact ? "p-4" : "p-5"}`}
    >
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
        <div className="flex items-center gap-2.5">
          {/* Animated risk dot */}
          <span className="relative flex h-3 w-3 shrink-0">
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-50 ${riskColor.dot}`}
            />
            <span className={`relative inline-flex rounded-full h-3 w-3 ${riskColor.dot}`} />
          </span>
          <h3 className="font-bold text-gray-900 text-sm">
            🤖 AI Moderation Analysis
          </h3>
          {moderation?.provider && moderation.provider !== "fallback" && (
            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 uppercase tracking-wide">
              {moderation.provider}
            </span>
          )}
        </div>
        {onRetry && !loading && (
          <button
            id="moderation-retry-btn"
            onClick={onRetry}
            className="text-xs text-gray-500 hover:text-gray-700 underline transition-colors"
          >
            Refresh Analysis
          </button>
        )}
      </div>

      {/* ── Loading Skeleton ── */}
      {loading && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <svg className="w-3.5 h-3.5 animate-spin text-purple-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Generating AI moderation analysis…
          </div>
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
          <Skeleton className="h-3 w-3/5" />
          <div className="flex gap-2 mt-2">
            <Skeleton className="h-6 w-24 rounded-full" />
            <Skeleton className="h-6 w-32 rounded-full" />
          </div>
        </div>
      )}

      {/* ── Error State ── */}
      {!loading && error && (
        <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl p-3">
          <span className="text-base mt-0.5">⚠️</span>
          <div>
            <p className="font-semibold">Analysis unavailable</p>
            <p className="text-xs mt-0.5 opacity-80">{error}</p>
            {onRetry && (
              <button onClick={onRetry} className="text-xs mt-1.5 text-red-600 underline hover:text-red-800">
                Try again
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── Result Content ── */}
      {!loading && moderation && (
        <div className="space-y-3">
          {/* Summary headline */}
          {moderation.summary && (
            <p className="text-xs font-semibold text-gray-600 italic leading-relaxed">
              "{moderation.summary}"
            </p>
          )}

          {/* Explanation (typewriter reveal) */}
          <div className="text-sm text-gray-800 leading-relaxed">
            {revealedExplanation}
            {revealedExplanation.length < (moderation.explanation?.length || 0) && (
              <span className="inline-block w-0.5 h-3.5 ml-0.5 bg-gray-700 animate-pulse align-middle" />
            )}
          </div>

          {/* Reasoning bullets */}
          {!compact && reasoningBullets.length > 0 && (
            <div className="space-y-1 pt-1">
              <p className="text-[10px] uppercase font-bold text-gray-400 tracking-widest">
                Escalation Signals
              </p>
              <ul className="space-y-1">
                {reasoningBullets.map((line, i) => (
                  <li
                    key={i}
                    className="text-xs text-gray-700 flex items-start gap-1.5 leading-snug"
                    style={{ animationDelay: `${i * 80}ms` }}
                  >
                    <span className="mt-0.5 shrink-0 text-gray-400">
                      {line.startsWith("•") ? "" : "•"}
                    </span>
                    {line.replace(/^•\s*/, "")}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Suggested action badge */}
          <div
            className={`flex items-center gap-2 mt-2 px-3 py-2 rounded-xl border ${action.bg} ${action.border} flex-wrap`}
          >
            <span
              id="moderation-action-badge"
              className={`font-bold text-xs px-2.5 py-1 rounded-full ${action.badge} uppercase tracking-wide whitespace-nowrap`}
            >
              {action.label}
            </span>
            <span className={`text-xs ${action.text} opacity-80`}>{action.description}</span>
          </div>
        </div>
      )}

      {/* ── Shimmer keyframe (injected once) ── */}
      <style>{`
        @keyframes shimmer {
          0% { background-position: 100% 0; }
          100% { background-position: -100% 0; }
        }
      `}</style>
    </div>
  );
}
