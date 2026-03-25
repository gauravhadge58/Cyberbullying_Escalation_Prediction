/**
 * API routes for the cyberbullying backend.
 * Proxies requests to the Python ML service and persists results to MongoDB.
 */
const express = require("express");
const axios = require("axios");
const FormData = require("form-data");
const router = express.Router();

const Message = require("../models/Message");
const Conversation = require("../models/Conversation");
const muteManager = require("../utils/muteManager");

const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

// ─────────────────────────────────────────────
// POST /api/predict
// Accepts JSON payload, forwards to ML service,
// stores results in MongoDB, broadcasts via WebSocket.
// ─────────────────────────────────────────────
router.post("/predict", async (req, res) => {
  try {
    const { messages } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "messages array required" });
    }

    // Check if any users are muted
    const mutedUsers = messages.filter(m => muteManager.isMuted(m.user_id));
    if (mutedUsers.length > 0) {
      const firstMuted = mutedUsers[0].user_id;
      return res.status(403).json({ 
        error: `User ${firstMuted} is temporarily muted for repeated violations.`,
        muted: true,
        remaining: muteManager.getRemainingMuteTime(firstMuted)
      });
    }

    // Enhance payload with historical context for better escalation prediction
    const conversationIds = [...new Set(messages.map(m => m.conversation_id))];
    const histories = await Message.find({ 
      conversationId: { $in: conversationIds } 
    }).sort({ timestamp: -1 }).limit(50).lean();

    // Merge history with new messages
    // Note: We use Map to handle potential duplicates if ID is already in DB
    const messageMap = new Map();
    histories.forEach(h => messageMap.set(h.messageId, {
      id: h.messageId,
      conversation_id: h.conversationId,
      user_id: h.userId,
      message: h.text,
      timestamp: h.timestamp.toISOString()
    }));
    
    messages.forEach(m => messageMap.set(m.id, m));
    const fullContext = Array.from(messageMap.values());

    // Forward full context to ML service
    const mlResponse = await axios.post(`${ML_URL}/predict`, { messages: fullContext });
    const { messages: mlMsgResults, conversations: convResults } = mlResponse.data;

    // We only want to return/save the NEW messages from the ML response
    const newMsgIds = new Set(messages.map(m => m.id));
    const msgResults = mlMsgResults.filter(m => newMsgIds.has(m.id));

    // Persist messages to MongoDB (upsert to avoid duplicates)
    const msgOps = msgResults.map((m) => ({
      updateOne: {
        filter: { messageId: m.id },
        update: {
          $set: {
            messageId: m.id,
            conversationId: m.conversation_id,
            userId: m.user_id,
            text: m.message,
            timestamp: new Date(m.timestamp),
            isBullying: m.is_bullying,
            toxicityScore: m.toxicity_score,
            confidence: m.confidence,
          },
        },
        upsert: true,
      },
    }));
    if (msgOps.length) await Message.bulkWrite(msgOps);

    // Persist conversations to MongoDB
    const convOps = convResults.map((c) => ({
      updateOne: {
        filter: { conversationId: c.conversation_id },
        update: {
          $set: {
            conversationId: c.conversation_id,
            escalationLevel: c.escalation_level,
            escalationScore: c.escalation_score,
            flagged: c.escalation_level === "HIGH",
            messageCount: c.message_count,
            features: c.features,
            lastUpdated: new Date(),
          },
        },
        upsert: true,
      },
    }));
    if (convOps.length) await Conversation.bulkWrite(convOps);

    // ── Automated Moderation (Auto-Mute) ──────────
    const penalizedUsers = new Set();

    // 1. Direct Toxicity Strike (Any message >= 0.95 toxicity)
    msgResults.forEach(m => {
      if (m.toxicity_score >= 0.95) {
        penalizedUsers.add(m.user_id);
      }
    });

    // 2. Room-Level Strike (Conversation is HIGH risk)
    convResults.forEach(c => {
      if (c.escalation_level === "HIGH") {
        msgResults
          .filter(m => m.conversation_id === c.conversation_id && m.is_bullying)
          .forEach(m => penalizedUsers.add(m.user_id));
      }
    });

    // Apply violations
    penalizedUsers.forEach(uid => {
      const newlyMuted = muteManager.addViolation(uid);
      if (newlyMuted) {
        console.log(`🚫 User ${uid} has been muted for 10 minutes.`);
      }
    });

    // Broadcast via WebSocket if app has ws server attached
    const wss = req.app.get("wss");
    if (wss) {
      const payload = JSON.stringify({
        type: "PREDICTION_UPDATE",
        data: { messages: msgResults, conversations: convResults },
      });
      wss.clients.forEach((client) => {
        if (client.readyState === 1) client.send(payload);
      });
    }

    res.json({ messages: msgResults, conversations: convResults });
  } catch (err) {
    console.error("Predict error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// POST /api/train
// Accepts multipart CSV upload, forwards to ML service.
// ─────────────────────────────────────────────
router.post("/train", async (req, res) => {
  try {
    if (!req.files || !req.files.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const file = req.files.file;
    const form = new FormData();
    form.append("file", file.data, {
      filename: file.name,
      contentType: file.mimetype,
    });

    const mlResponse = await axios.post(`${ML_URL}/train`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
    });

    res.json(mlResponse.data);
  } catch (err) {
    console.error("Train error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// GET /api/conversations
// Fetches stored conversations from MongoDB.
// ─────────────────────────────────────────────
router.get("/conversations", async (req, res) => {
  try {
    const conversations = await Conversation.find()
      .sort({ lastUpdated: -1 })
      .limit(100)
      .lean();

    // For each conversation, attach its messages
    const enriched = await Promise.all(
      conversations.map(async (conv) => {
        const msgs = await Message.find({ conversationId: conv.conversationId })
          .sort({ timestamp: 1 })
          .lean();
        return { ...conv, messages: msgs };
      })
    );

    res.json({ conversations: enriched, count: enriched.length });
  } catch (err) {
    console.error("Conversations error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// GET /api/stats
// Returns aggregate analytics from MongoDB.
// ─────────────────────────────────────────────
router.get("/stats", async (req, res) => {
  try {
    const totalMessages = await Message.countDocuments();
    const bullyingCount = await Message.countDocuments({ isBullying: true });
    const bullyingPct = totalMessages > 0 ? ((bullyingCount / totalMessages) * 100).toFixed(1) : 0;

    const escalationDist = await Conversation.aggregate([
      { $group: { _id: "$escalationLevel", count: { $sum: 1 } } },
    ]);

    const distMap = { LOW: 0, MEDIUM: 0, HIGH: 0 };
    escalationDist.forEach((e) => { if (e._id) distMap[e._id] = e.count; });

    const highConversations = await Conversation.find({ escalationLevel: "HIGH" })
      .sort({ lastUpdated: -1 })
      .limit(10)
      .lean();

    // Toxicity over time — last 50 messages sorted by timestamp
    const recentMsgs = await Message.find()
      .sort({ timestamp: -1 })
      .limit(50)
      .select("timestamp toxicityScore conversationId")
      .lean();

    const toxicityOverTime = recentMsgs.reverse().map((m) => ({
      timestamp: m.timestamp,
      toxicity: m.toxicityScore,
      conversationId: m.conversationId,
    }));

    res.json({
      totalMessages,
      bullyingCount,
      nonBullyingCount: totalMessages - bullyingCount,
      bullyingPercentage: parseFloat(bullyingPct),
      escalationDistribution: distMap,
      highEscalationConversations: highConversations,
      toxicityOverTime,
    });
  } catch (err) {
    console.error("Stats error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// DELETE /api/conversations
// Clears all messages and conversations from MongoDB.
// ─────────────────────────────────────────────
router.delete("/conversations", async (req, res) => {
  try {
    await Message.deleteMany({});
    await Conversation.deleteMany({});
    
    // Broadcast update via WebSocket
    const wss = req.app.get("wss");
    if (wss) {
      wss.broadcast({ type: "DATA_CLEARED", message: "All chat data has been reset." });
    }
    
    res.json({ message: "All messages and conversations cleared successfully" });
  } catch (err) {
    console.error("Clear error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// DELETE /api/conversations/:id
// Deletes a specific conversation and its messages.
// ─────────────────────────────────────────────
router.delete("/conversations/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await Message.deleteMany({ conversationId: id });
    await Conversation.deleteOne({ conversationId: id });
    
    // Broadcast update via WebSocket
    const wss = req.app.get("wss");
    if (wss) {
      wss.broadcast({ 
        type: "CONVERSATION_DELETED", 
        data: { conversationId: id },
        message: `Conversation ${id} has been cleared.` 
      });
    }
    
    res.json({ message: `Conversation ${id} cleared successfully` });
  } catch (err) {
    console.error("Clear room error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
