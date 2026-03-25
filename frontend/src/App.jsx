import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Monitor from "./pages/Monitor";
import Analytics from "./pages/Analytics";
import Upload from "./pages/Upload";
import Simulator from "./pages/Simulator";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="monitor" element={<Monitor />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="upload" element={<Upload />} />
        <Route path="simulator" element={<Simulator />} />
      </Route>
    </Routes>
  );
}


