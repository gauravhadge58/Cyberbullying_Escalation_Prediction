import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
    
    // Polling fallback if WebSocket isn't used
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const data = await api.getStats();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch stats", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="text-center mt-20 text-gray-500 animate-pulse">Loading dashboard...</div>;
  if (!stats) return <div className="text-center mt-20 text-red-500">Failed to load data</div>;

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900">System Dashboard</h2>
        <p className="text-gray-500 text-sm">Overview of recent activity and risk levels.</p>
      </div>
      {/* Active Alerts */}
      {stats.highEscalationConversations?.length > 0 && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg shadow-sm">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-red-800 font-bold flex items-center gap-2">
                <span>⚠️</span> High Escalation Alerts ({stats.highEscalationConversations.length})
              </h3>
              <p className="text-red-600 text-sm mt-1">
                Conversations require immediate review.
              </p>
            </div>
            <Link to="/monitor" className="bg-red-100 hover:bg-red-200 text-red-800 font-medium px-4 py-2 rounded-lg transition-colors text-sm">
              Review Now
            </Link>
          </div>
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Total Messages" value={stats.totalMessages} icon="💬" color="blue" />
        <StatCard title="Bullying Detected" value={stats.bullyingCount} sub={`${stats.bullyingPercentage}%`} icon="🚨" color="red" />
        <StatCard title="Safe Messages" value={stats.nonBullyingCount} icon="✅" color="green" />
        <StatCard title="High Risk Chats" value={stats.escalationDistribution?.HIGH || 0} icon="🔥" color="orange" />
      </div>

      {/* Recent High-Risk Conversations */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 bg-gray-50/50 flex justify-between items-center">
          <h2 className="font-semibold text-gray-800">Recent High-Risk Conversations</h2>
          <Link to="/monitor" className="text-sm text-primary-600 hover:text-primary-700 font-medium">View All &rarr;</Link>
        </div>
        <div className="divide-y divide-gray-100">
          {stats.highEscalationConversations?.slice(0, 5).map((conv) => (
            <div key={conv.conversationId} className="px-6 py-4 hover:bg-gray-50 transition-colors flex justify-between items-center">
              <div>
                <p className="font-medium text-gray-900">{conv.conversationId}</p>
                <p className="text-sm text-gray-500 mt-1">Score: {conv.escalationScore} / 10</p>
              </div>
              <span className="px-3 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">
                HIGH RISK
              </span>
            </div>
          ))}
          {stats.highEscalationConversations?.length === 0 && (
            <div className="px-6 py-8 text-center text-gray-500 text-sm">
              No high-risk conversations detected recently.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, sub, icon, color }) {
  const colorMap = {
    blue: "bg-blue-50 text-blue-600",
    red: "bg-red-50 text-red-600",
    green: "bg-green-50 text-green-600",
    orange: "bg-orange-50 text-orange-600",
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-center gap-4 hover:-translate-y-1 transition-transform duration-200">
      <div className={`w-14 h-14 rounded-full flex items-center justify-center text-2xl ${colorMap[color]}`}>
        {icon}
      </div>
      <div>
        <p className="text-sm font-medium text-gray-500">{title}</p>
        <div className="flex items-baseline gap-2 mt-1">
          <h4 className="text-2xl font-bold text-gray-900">{value}</h4>
          {sub && <span className="text-sm font-semibold text-red-500">{sub}</span>}
        </div>
      </div>
    </div>
  );
}
