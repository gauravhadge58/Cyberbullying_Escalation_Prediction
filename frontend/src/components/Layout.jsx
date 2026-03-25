import { Link, Outlet, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

const navItems = [
  { path: "/dashboard", label: "Dashboard", icon: "📊" },
  { path: "/monitor", label: "Live Monitor", icon: "👁️" },
  { path: "/analytics", label: "Analytics", icon: "📈" },
  { path: "/simulator", label: "Test Chat", icon: "💬" },
];

export default function Layout() {
  const location = useLocation();
  const [socketStatus, setSocketStatus] = useState("Connecting...");

  useEffect(() => {
    // Connect to WebSocket on port 5000 (Backend)
    const wsUrl = import.meta.env.VITE_WS_URL || "ws://localhost:5000";
    let ws;
    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => setSocketStatus("Connected");
      ws.onclose = () => setSocketStatus("Disconnected");
      ws.onerror = () => setSocketStatus("Error");
    } catch (err) {
      setSocketStatus("Failed");
    }
    return () => {
      if (ws) ws.close();
    };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col transition-all duration-300">
        <div className="h-16 flex items-center px-6 border-b border-gray-200 gap-3">
          <span className="text-2xl">🛡️</span>
          <h1 className="text-xl font-bold bg-gradient-to-r from-primary-600 to-primary-400 bg-clip-text text-transparent">
            Defend
          </h1>
        </div>

        <nav className="flex-1 overflow-y-auto py-6 px-4 space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname.startsWith(item.path);
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                  isActive
                    ? "bg-primary-50 text-primary-700 font-semibold shadow-sm"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`}
              >
                <span className="text-lg">{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center gap-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-lg">
            <div
              className={`w-2 h-2 rounded-full ${
                socketStatus === "Connected" ? "bg-success" : "bg-danger"
              }`}
            />
            WS: {socketStatus}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <header className="h-16 bg-white border-b border-gray-200 flex items-center px-8 shadow-sm z-10">
          <h2 className="text-lg font-semibold text-gray-800 capitalize">
            {location.pathname.substring(1) || "Dashboard"}
          </h2>
        </header>
        <div className="flex-1 overflow-y-auto p-8 relative">
          <div className="max-w-7xl mx-auto h-full animate-fade-in">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
