import { useState, useRef, useEffect } from "react";
import { api } from "../api";

export default function Training() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const fileInputRef = useRef(null);

  const [trainingProgress, setTrainingProgress] = useState(null);
  const pollIntervalRef = useRef(null);

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  // Model registry state
  const [models, setModels] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [registryLoading, setRegistryLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(null); // model id being acted on

  // ─── Load Registry ──────────────────────────────────
  const loadModels = async () => {
    try {
      setRegistryLoading(true);
      const data = await api.listModels();
      setModels(data.models || []);
      setActiveId(data.active_id);
    } catch (err) {
      console.error("Failed to load models:", err);
    } finally {
      setRegistryLoading(false);
    }
  };

  useEffect(() => { loadModels(); }, []);

  // ─── Actions ────────────────────────────────────────
  const handleFile = (f) => {
    if (f && f.name.toLowerCase().endsWith(".csv")) {
      setFile(f);
      setStatus("idle");
      setResult(null);
    } else {
      alert("Please select a valid .csv file.");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleTrain = async () => {
    if (!file) return;
    setStatus("loading");
    setResult(null);
    setErrorMsg("");
    setTrainingProgress(null);

    // Start polling progress
    pollIntervalRef.current = setInterval(async () => {
      try {
        const prog = await api.getTrainingProgress();
        if (prog && prog.status !== "idle") {
          setTrainingProgress(prog);
        }
      } catch (e) {
        // ignore fetch errors during polling gently
      }
    }, 1000);

    try {
      const data = await api.trainModels(file);
      setResult(data);
      setStatus("success");
      await loadModels(); // refresh registry
    } catch (err) {
      setErrorMsg(err?.response?.data?.detail || err.message || "Training failed.");
      setStatus("error");
    } finally {
      clearInterval(pollIntervalRef.current);
    }
  };

  const handleActivate = async (modelId) => {
    setActionLoading(modelId);
    try {
      await api.activateModel(modelId);
      setActiveId(modelId);
    } catch (err) {
      alert(`Failed to activate: ${err.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (modelId) => {
    if (!window.confirm(`Delete model "${modelId}"? This cannot be undone.`)) return;
    setActionLoading(modelId);
    try {
      const data = await api.deleteModel(modelId);
      setActiveId(data.active_id);
      setModels((prev) => prev.filter((m) => m.id !== modelId));
    } catch (err) {
      alert(`Failed to delete: ${err.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const formatAccuracy = (val) =>
    val !== undefined && val !== null ? `${(val * 100).toFixed(1)}%` : "N/A";

  // ─── Pipeline Visualizer ────────────────────────────
  const renderPipeline = () => {
    if (!trainingProgress) return (
        <div className="mt-8 flex flex-col items-center justify-center p-8 gap-3 border border-gray-100 rounded-2xl bg-gray-50/50">
             <div className="animate-spin h-8 w-8 border-4 border-gray-200 border-t-primary-600 rounded-full"></div>
             <p className="text-gray-500 font-medium animate-pulse">Initializing Training Engine...</p>
        </div>
    );

    const phases = [
      { id: "bert", label: "BERT Semantic Analysis", icon: "🤖" },
      { id: "lstm", label: "LSTM Sequential Model", icon: "🧠" },
      { id: "rf",   label: "Random Forest Meta-Learner", icon: "🌲" }
    ];

    const currentIdx = phases.findIndex(p => p.id === trainingProgress.phase);
    // If phase is not found exactly, treat as early init or complete
    const activeIdx = currentIdx >= 0 ? currentIdx : 0;

    return (
      <div className="mt-8 border border-gray-100 rounded-2xl bg-gray-50/50 p-6 shadow-inner animate-fade-in relative overflow-hidden">
        {/* Animated background glow */}
        <div className="absolute top-0 right-1/4 w-1/2 h-full bg-primary-400/5 blur-[80px] -z-10 animate-pulse-slow"></div>

        <h3 className="text-lg font-bold text-gray-800 mb-6 flex items-center gap-2">
            <svg className="animate-spin h-5 w-5 text-primary-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Training in Progress
        </h3>
        
        <div className="space-y-6">
          {phases.map((phase, i) => {
            const isActive = phase.id === trainingProgress.phase;
            // Phase is complete if we are past it, or if final status is success
            const isCompleted = currentIdx > i || trainingProgress.status === "success" || trainingProgress.phase === "complete";

            return (
              <div 
                key={phase.id} 
                className={`relative flex items-center gap-4 p-4 rounded-xl transition-all duration-500 ${
                  isActive 
                  ? "bg-white shadow-md border-l-4 border-l-primary-500 transform scale-[1.02]" 
                  : isCompleted 
                    ? "bg-green-50/30 border border-transparent opacity-90" 
                    : "opacity-40 grayscale"
                }`}
              >
                {/* Visual Line connector */}
                {i < phases.length - 1 && (
                  <div className={`absolute left-[38px] top-14 w-0.5 h-8 z-0 ${isCompleted ? 'bg-green-400' : 'bg-gray-200'}`}></div>
                )}
                
                <div className={`w-12 h-12 flex items-center justify-center rounded-full text-2xl z-10 shrink-0 shadow-sm transition-colors duration-500 ${
                    isActive ? "bg-primary-100 ring-4 ring-primary-50 animate-pulse-slow text-primary-600" 
                    : isCompleted ? "bg-green-100 text-green-600 ring-4 ring-green-50" 
                    : "bg-gray-100 text-gray-400"
                }`}>
                  {isCompleted ? "✓" : phase.icon}
                </div>
                
                <div className="flex-1 w-full min-w-0">
                  <div className="flex justify-between items-baseline mb-1">
                    <h4 className={`font-bold transition-colors duration-500 ${isActive ? "text-primary-800" : isCompleted ? "text-green-800" : "text-gray-600"}`}>
                      {phase.label}
                    </h4>
                    {isActive && trainingProgress.total > 0 && (
                      <span className="text-xs font-mono font-medium text-primary-700 bg-primary-50 px-2 py-0.5 rounded-full ring-1 ring-primary-200">
                        {trainingProgress.current} / {trainingProgress.total} {phase.id === 'lstm' ? 'Epochs' : ''}
                      </span>
                    )}
                  </div>
                  
                  {isActive ? (
                    <div className="space-y-3">
                       <p className="text-sm text-gray-600 italic truncate">{trainingProgress.message}</p>
                       {/* Progress Bar */}
                       {trainingProgress.total > 0 && (
                          <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden shadow-inner">
                             <div 
                               className="h-full bg-primary-500 rounded-full transition-all duration-300 ease-out relative"
                               style={{ width: `${Math.min(100, (trainingProgress.current / trainingProgress.total) * 100)}%` }}
                             >
                               <div className="absolute inset-0 bg-white/20 w-full animate-shine"></div>
                             </div>
                          </div>
                       )}
                    </div>
                  ) : isCompleted ? (
                    <p className="text-xs font-semibold text-green-600">Phase Complete</p>
                  ) : (
                    <p className="text-xs font-medium text-gray-400">Waiting...</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // ─── Render ─────────────────────────────────────────
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">Train AI Models</h1>
        <p className="mt-2 text-gray-500 font-medium">Upload a labelled dataset to re-train the Cyberbullying detection and escalation models.</p>
      </div>

      {/* Upload Section */}
      <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 relative">
        <input 
          type="file" 
          accept=".csv" 
          ref={fileInputRef} 
          className="hidden" 
          onChange={(e) => handleFile(e.target.files[0])} 
        />
        
        <div
          className={`border-2 border-dashed rounded-xl p-10 text-center transition-all ${
            isDragging ? "border-primary-500 bg-primary-50 scale-[1.02]" 
            : file ? "border-green-400 bg-green-50"
            : "border-gray-300 hover:border-primary-400 hover:bg-gray-50/50"
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {file ? (
            <div className="text-green-700">
              <span className="text-4xl block mb-3">📄</span>
              <p className="font-semibold text-lg">{file.name}</p>
              <p className="text-sm opacity-80 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              <button 
                onClick={() => setFile(null)} 
                className="mt-4 text-sm text-green-600 underline font-medium hover:text-green-800"
              >
                Remove file
              </button>
            </div>
          ) : (
            <div>
               <span className="text-4xl block mb-3 opacity-60">📥</span>
               <p className="text-gray-600 font-medium text-lg">
                <button 
                  onClick={() => fileInputRef.current?.click()} 
                  className="text-primary-600 font-semibold hover:underline"
                >
                  Click to upload
                </button>{" "}
                or drag and drop
              </p>
              <p className="text-sm text-gray-400 mt-2">CSV files only. Must include 'message' and 'label' columns.</p>
            </div>
          )}
        </div>

        {/* Train State */}
        {status === "loading" ? (
          renderPipeline()
        ) : (
          <button
            onClick={handleTrain}
            disabled={!file}
            className="mt-5 w-full py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-all shadow-sm active:scale-[0.99] flex items-center justify-center gap-2"
          >
            🚀 Train & Save New Model
          </button>
        )}
      </div>

      {/* ═══ ERROR ═══════════════════════════════════ */}
      {status === "error" && (
        <div className="bg-red-50 border border-red-200 rounded-2xl p-5 text-red-700">
          <p className="font-semibold mb-1">❌ Training Failed</p>
          <p className="text-sm font-mono">{errorMsg}</p>
        </div>
      )}

      {/* ═══ RESULTS ═════════════════════════════════ */}
      {status === "success" && result && (
        <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-green-200 mt-8 animate-slide-up">
          <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            <span className="text-green-500">✅</span> Training Complete!
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 rounded-xl p-5 border border-gray-100">
              <h4 className="font-semibold text-gray-700 mb-3 border-b pb-2">BERT Detection Model</h4>
              <p className="text-sm flex justify-between"><span className="text-gray-500">Train Size:</span> <span className="font-mono">{result.detection?.train_size || 'N/A'}</span></p>
              <p className="text-sm flex justify-between"><span className="text-gray-500">Accuracy:</span> <span className="font-mono">{formatAccuracy(result.detection?.accuracy)}</span></p>
            </div>
            
            <div className="bg-gray-50 rounded-xl p-5 border border-gray-100">
              <h4 className="font-semibold text-gray-700 mb-3 border-b pb-2">Escalation Model</h4>
              <p className="text-sm flex justify-between"><span className="text-gray-500">Train Size:</span> <span className="font-mono">{result.escalation?.train_size || 'N/A'}</span></p>
              <p className="text-sm flex justify-between"><span className="text-gray-500">Accuracy:</span> <span className="font-mono">{formatAccuracy(result.escalation?.accuracy)}</span></p>
            </div>
          </div>
        </div>
      )}

      {/* ═══ REGISTRY ════════════════════════════════ */}
      <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100">
         <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
             <span className="text-2xl">💾</span> Model Registry
         </h2>
         
         {registryLoading ? (
            <div className="text-center py-8 text-gray-500">Loading registry...</div>
         ) : models.length === 0 ? (
            <div className="text-center py-8 text-gray-500 bg-gray-50 rounded-xl border border-dashed border-gray-200">
                No custom models trained yet. The default pre-trained transformer is active.
            </div>
         ) : (
            <div className="space-y-4">
               {models.map(m => {
                  const isActive = m.id === activeId;
                  const isActing = actionLoading === m.id;
                  
                  return (
                    <div key={m.id} className={`flex flex-col md:flex-row items-center justify-between p-4 rounded-xl border transition-all ${isActive ? 'bg-primary-50 border-primary-200 shadow-sm ring-1 ring-primary-500' : 'bg-white border-gray-200 hover:border-gray-300'}`}>
                       <div className="flex-1 w-full mb-4 md:mb-0">
                          <div className="flex items-center gap-2 mb-1">
                             <h4 className="font-bold text-gray-800">{m.label}</h4>
                             {isActive && <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full font-medium ring-1 ring-green-400">ACTIVE PROD MODEL</span>}
                          </div>
                          <div className="flex items-center gap-3 text-sm text-gray-500 font-mono">
                              <span>ID: {m.id}</span> • 
                              <span>{new Date(m.trained_at).toLocaleString()}</span>
                          </div>
                          
                          <div className="mt-3 flex gap-2">
                              {m.rf_available && <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-md text-xs font-semibold">🌲 Random Forest</span>}
                              {m.lstm_available && <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-md text-xs font-semibold">🧠 LSTM Deep Learning</span>}
                              {!m.rf_available && !m.lstm_available && <span className="px-2 py-1 bg-gray-100 text-gray-400 rounded-md text-xs font-semibold">⚠️ Legacy Missing</span>}
                          </div>
                       </div>
                       
                       <div className="flex items-center gap-2 w-full md:w-auto">
                          {!isActive && (
                            <button 
                               onClick={() => handleActivate(m.id)}
                               disabled={isActing}
                               className="flex-1 md:flex-none px-4 py-2 bg-white text-primary-600 border border-primary-200 hover:bg-primary-50 rounded-lg font-medium transition-colors disabled:opacity-50"
                            >
                               {isActing ? "Activating..." : "Set as Active"}
                            </button>
                          )}
                          <button 
                             onClick={() => handleDelete(m.id)}
                             disabled={isActing || isActive}
                             className="flex-1 md:flex-none px-4 py-2 bg-white text-red-600 border border-red-200 hover:bg-red-50 rounded-lg font-medium transition-colors disabled:opacity-50"
                             title={isActive ? "Cannot delete the active model" : "Delete model"}
                          >
                             Delete
                          </button>
                       </div>
                    </div>
                  );
               })}
            </div>
         )}
      </div>

    </div>
  );
}
