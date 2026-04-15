/**
 * Gemini-powered chatbot utility for the CERDS cyberbullying platform.
 * Builds a context-rich system prompt from live DB stats and handles
 * multi-turn conversation history.
 */
const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

/**
 * Build the system prompt with live project stats injected.
 */
function buildSystemPrompt(stats) {
  const dist = stats?.escalationDistribution || { LOW: 0, MEDIUM: 0, HIGH: 0 };
  const total = stats?.totalMessages || 0;
  const bullying = stats?.bullyingCount || 0;
  const bullyPct = stats?.bullyingPercentage || 0;

  return `You are CERDS-AI, a helpful assistant embedded in the CERDS platform — 
Cyberbullying Escalation Risk Detection System.

## About CERDS
CERDS is an AI-powered platform that detects and predicts cyberbullying escalation in online conversations.
It uses a hierarchical ML pipeline:
  1. **BERT-based Toxicity Detection** — classifies each message as bullying / non-bullying and assigns a toxicity score (0–1).
  2. **LSTM Sequential Model** — captures temporal patterns across a conversation thread.
  3. **Random Forest Meta-Learner** — combines BERT + LSTM features to predict escalation level (LOW / MEDIUM / HIGH).

## Tech Stack
- Frontend: React + Vite + TailwindCSS
- Backend: Node.js / Express + MongoDB (stores all messages and conversations)
- ML Service: Python / FastAPI (runs BERT, LSTM, Random Forest)
- Real-time: WebSocket for live monitor updates

## Pages in the App
- **Dashboard** — high-level KPIs, recent alerts, escalation distribution
- **Live Monitor** — real-time feed of all flagged conversations
- **Analytics** — charts: toxicity over time, escalation breakdown, bully ratio
- **Test Chat (Simulator)** — type messages and see live ML predictions
- **Model Training** — upload a CSV dataset and retrain the RF escalation model

## Live Platform Stats (current data)
- Total messages analysed: ${total}
- Bullying messages detected: ${bullying} (${bullyPct}%)
- Escalation breakdown — LOW: ${dist.LOW}, MEDIUM: ${dist.MEDIUM}, HIGH: ${dist.HIGH}

## Your Role
Answer questions about:
- How the ML pipeline works (BERT, LSTM, Random Forest, feature extraction)
- How to use the platform (navigate pages, train a model, test the simulator)
- Interpreting results (what escalation scores mean, toxicity thresholds)
- The research paper behind CERDS (hierarchical ensemble for cyberbullying detection)
- General cyberbullying awareness and online safety advice

Keep answers concise, helpful, and friendly. If the user asks something unrelated to cyberbullying, 
online safety, or the CERDS platform, politely redirect them.`;
}

/**
 * Send a chat message to Gemini and get a response.
 * @param {string} userMessage - The latest user message
 * @param {Array}  history     - Previous [{role, parts: [{text}]}] turns
 * @param {object} stats       - Live stats from MongoDB for context injection
 * @returns {Promise<{reply: string, history: Array}>}
 */
async function chat(userMessage, history = [], stats = {}) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured in the backend .env file.");
  }

  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash-preview-04-17",
    systemInstruction: buildSystemPrompt(stats),
  });

  const chatSession = model.startChat({
    history,
    generationConfig: {
      maxOutputTokens: 600,
      temperature: 0.7,
    },
  });

  const result = await chatSession.sendMessage(userMessage);
  const reply = result.response.text();

  // Append this turn to history for next call
  const updatedHistory = [
    ...history,
    { role: "user", parts: [{ text: userMessage }] },
    { role: "model", parts: [{ text: reply }] },
  ];

  return { reply, history: updatedHistory };
}

module.exports = { chat };
