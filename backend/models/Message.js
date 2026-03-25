/**
 * Mongoose schema for a Message document.
 * Stores individual messages with detection results.
 */
const mongoose = require("mongoose");

const MessageSchema = new mongoose.Schema(
  {
    messageId: { type: String, required: true, unique: true },
    conversationId: { type: String, required: true, index: true },
    userId: { type: String, default: "unknown" },
    text: { type: String, required: true },
    timestamp: { type: Date, default: Date.now },
    isBullying: { type: Boolean, default: false },
    toxicityScore: { type: Number, default: 0 },
    confidence: { type: Number, default: 0 },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Message", MessageSchema);
