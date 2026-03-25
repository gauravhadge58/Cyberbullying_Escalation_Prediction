/**
 * Mute Manager Utility
 * Tracks user violations and handles temporary mutes.
 */

const VIOLATION_THRESHOLD = 3;
const MUTE_DURATION_MS = 10 * 60 * 1000; // 10 minutes

// In-memory status tracking (will reset on server restart)
const userStats = new Map(); // userId -> { violations: number, mutedUntil: timestamp | null }

const muteManager = {
  /**
   * Check if a user is currently muted.
   */
  isMuted: (userId) => {
    const stats = userStats.get(userId);
    if (!stats || !stats.mutedUntil) return false;
    
    // Check if mute has expired
    if (Date.now() > stats.mutedUntil) {
      stats.mutedUntil = null;
      stats.violations = 0; // Reset violations on expiry
      return false;
    }
    return true;
  },

  /**
   * Get remaining mute time in seconds.
   */
  getRemainingMuteTime: (userId) => {
    const stats = userStats.get(userId);
    if (!stats || !stats.mutedUntil) return 0;
    const remaining = Math.max(0, Math.ceil((stats.mutedUntil - Date.now()) / 1000));
    return remaining;
  },

  /**
   * Track a new violation for a user.
   * Returns true if the user was just muted.
   */
  addViolation: (userId) => {
    let stats = userStats.get(userId);
    if (!stats) {
      stats = { violations: 0, mutedUntil: null };
      userStats.set(userId, stats);
    }

    stats.violations += 1;

    if (stats.violations >= VIOLATION_THRESHOLD) {
      stats.mutedUntil = Date.now() + MUTE_DURATION_MS;
      return true; // Newly muted
    }
    return false;
  },

  /**
   * Manual unmute if needed.
   */
  unmute: (userId) => {
    userStats.set(userId, { violations: 0, mutedUntil: null });
  }
};

module.exports = muteManager;
