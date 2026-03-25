/**
 * MongoDB connection configuration using Mongoose.
 * Reads MONGO_URI from environment variables.
 */
const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(
      process.env.MONGO_URI || "mongodb://localhost:27017/cyberbullying_db",
      {
        serverSelectionTimeoutMS: 5000,
      }
    );
    console.log(`✅ MongoDB connected: ${conn.connection.host}`);
  } catch (error) {
    console.error(`❌ MongoDB connection error: ${error.message}`);
    // Don't exit — allow app to run without DB in development
  }
};

module.exports = connectDB;
