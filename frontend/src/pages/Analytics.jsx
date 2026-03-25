import { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import { api } from "../api";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function Analytics() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const data = await api.getStats();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch analytics", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="text-center mt-20 text-gray-500 animate-pulse">Loading analytics...</div>;
  if (!stats) return <div className="text-center mt-20 text-red-500">Failed to load data</div>;

  // Prepare Toxicity Over Time (Line Chart) data
  const toxicityLabels = stats.toxicityOverTime?.map(d => new Date(d.timestamp).toLocaleTimeString()) || [];
  const toxicityData = stats.toxicityOverTime?.map(d => d.toxicity) || [];

  const lineChartData = {
    labels: toxicityLabels,
    datasets: [
      {
        label: "Toxicity Score (Last 50 Msgs)",
        data: toxicityData,
        borderColor: "rgb(239, 68, 68)", // red-500
        backgroundColor: "rgba(239, 68, 68, 0.1)",
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 2,
      },
    ],
  };

  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
    },
    scales: {
      y: { beginAtZero: true, max: 1 },
      x: {
        ticks: { maxTicksLimit: 10 },
      }
    },
  };

  // Prepare Escalation Distribution (Bar Chart) data
  const dist = stats.escalationDistribution || { LOW: 0, MEDIUM: 0, HIGH: 0 };
  const barChartData = {
    labels: ["Low Risk", "Medium Risk", "High Risk"],
    datasets: [
      {
        label: "Number of Conversations",
        data: [dist.LOW || 0, dist.MEDIUM || 0, dist.HIGH || 0],
        backgroundColor: [
          "rgba(16, 185, 129, 0.6)", // green
          "rgba(245, 158, 11, 0.6)", // orange
          "rgba(239, 68, 68, 0.6)",  // red
        ],
        borderColor: [
          "rgb(16, 185, 129)",
          "rgb(245, 158, 11)",
          "rgb(239, 68, 68)",
        ],
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      y: { beginAtZero: true, ticks: { stepSize: 1 } },
    },
  };

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900">System Analytics</h2>
        <p className="text-gray-500 text-sm mt-1">Macro trends across all monitored conversations.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Toxicity Trend Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-96">
          <h3 className="font-semibold text-gray-800 mb-4">Message Toxicity Trend</h3>
          <div className="h-72">
            <Line data={lineChartData} options={lineChartOptions} />
          </div>
        </div>

        {/* Escalation Distribution Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-96">
          <h3 className="font-semibold text-gray-800 mb-4">Escalation Levels Distribution</h3>
          <div className="h-72">
            <Bar data={barChartData} options={barChartOptions} />
          </div>
        </div>
        
      </div>
      
      {/* Platform Summary Text */}
      <div className="bg-primary-50 p-6 rounded-xl border border-primary-100 mt-6">
        <h4 className="font-semibold text-primary-900 mb-2">Platform Summary</h4>
        <p className="text-sm text-primary-800">
          The system has analyzed <strong>{stats.totalMessages}</strong> messages across all active conversations. 
          Currently, <strong>{stats.bullyingPercentage}%</strong> of traffic is flagged as toxic or bullying. 
          There are <strong>{dist.HIGH || 0}</strong> conversations requiring immediate moderation intervention.
        </p>
      </div>
    </div>
  );
}
