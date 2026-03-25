/**
 * Mongoose schema for a Conversation document.
 * Stores grouped conversations with escalation predictions.
 */
const mongoose = require("mongoose");

const ConversationSchema = new mongoose.Schema(
  {
    conversationId: { type: String, required: true, unique: true, index: true },
    escalationLevel: {
      type: String,
      enum: ["LOW", "MEDIUM", "HIGH"],
      default: "LOW",
    },
    escalationScore: { type: Number, default: 0 },
    flagged: { type: Boolean, default: false },
    messageCount: { type: Number, default: 0 },
    bullyingCount: { type: Number, default: 0 },
    features: { type: mongoose.Schema.Types.Mixed, default: {} },
    lastUpdated: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Conversation", ConversationSchema);
