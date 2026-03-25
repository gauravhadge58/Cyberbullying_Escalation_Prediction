/**
 * Main Express server with:
 * - REST API routes
 * - WebSocket server for real-time updates
 * - MongoDB connection
 * - File upload middleware
 */
require("dotenv").config();
const http = require("http");
const express = require("express");
const cors = require("cors");
const fileUpload = require("express-fileupload");
const { WebSocketServer } = require("ws");

const connectDB = require("./config/db");
const apiRoutes = require("./routes/api");

const app = express();
const PORT = process.env.PORT || 5000;

// ── Middleware ─────────────────────────────────
app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(fileUpload({ limits: { fileSize: 50 * 1024 * 1024 } })); // 50 MB max

// ── Database ───────────────────────────────────
connectDB();

// ── Routes ─────────────────────────────────────
app.use("/api", apiRoutes);

app.get("/health", (req, res) => res.json({ status: "ok", service: "backend" }));

// ── HTTP + WebSocket Server ────────────────────
const server = http.createServer(app);

// Attach WebSocket server to the same HTTP server
const wss = new WebSocketServer({ server });
app.set("wss", wss);

wss.on("connection", (ws) => {
  console.log("🔌 WebSocket client connected");
  ws.send(JSON.stringify({ type: "CONNECTED", message: "Real-time updates active" }));

  ws.on("close", () => console.log("🔌 WebSocket client disconnected"));
  ws.on("error", (err) => console.error("WS error:", err.message));
});

// Broadcast helper (can be imported by routes)
wss.broadcast = (data) => {
  const payload = typeof data === "string" ? data : JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client.readyState === 1) client.send(payload);
  });
};

server.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`🔗 WebSocket server ready on ws://localhost:${PORT}`);
});
