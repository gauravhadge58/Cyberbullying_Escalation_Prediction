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
    if (f && f.name.endsWith(".csv")) {
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
      { id: "bert", label: "BERT Detection", icon: "🤖" },
      { id: "rf", label: "Random Forest Ensemble", icon: "🌲" },
      { id: "lstm", label: "LSTM Deep Learning", icon: "🧠" }
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
        <h1 className="text-2xl font-bold text-gray-900">Model Training & Management</h1>
        <p className="text-gray-500 text-sm mt-1">
          Train new models, view saved versions, and switch the active model used for predictions.
        </p>
      </div>

      {/* ═══ MODEL REGISTRY ═══════════════════════════ */}
      <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-gray-800">📦 Saved Models</h2>
          <button
            onClick={loadModels}
            disabled={registryLoading}
            className="text-xs text-primary-600 hover:text-primary-700 font-medium flex items-center gap-1"
          >
            {registryLoading ? "Loading..." : "🔄 Refresh"}
          </button>
        </div>

        {registryLoading ? (
          <div className="text-center py-8 text-gray-400">
            <div className="animate-spin inline-block w-6 h-6 border-2 border-gray-300 border-t-primary-500 rounded-full mb-2" />
            <p className="text-sm">Loading model registry...</p>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <div className="text-4xl mb-2">📭</div>
            <p className="font-medium">No models found</p>
            <p className="text-sm mt-1">Train your first model by uploading a CSV below.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {models.map((m) => {
              const isActive = m.id === activeId;
              const isActing = actionLoading === m.id;
              return (
                <div
                  key={m.id}
                  className={`border rounded-xl p-4 transition-all ${
                    isActive
                      ? "border-primary-400 bg-primary-50 ring-1 ring-primary-200"
                      : "border-gray-200 bg-gray-50 hover:bg-white"
                  }`}
                >
                  <div className="flex flex-col md:flex-row md:items-center gap-3">
                    {/* Left: Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className="font-semibold text-gray-800 text-sm truncate">
                          {m.label || m.id}
                        </span>
                        {isActive && (
                          <span className="text-[10px] font-bold uppercase bg-primary-500 text-white px-2 py-0.5 rounded-full">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
                        <span>🕐 {m.trained_at || "Unknown"}</span>
                        {m.rf_available && (
                          <span className="text-green-600 font-medium">✅ Random Forest</span>
                        )}
                        {m.lstm_available && (
                          <span className="text-purple-600 font-medium">✅ LSTM</span>
                        )}
                        {m.lstm_accuracy != null && (
                          <span>🎯 LSTM Acc: {formatAccuracy(m.lstm_accuracy)}</span>
                        )}
                        {m.num_conversations != null && (
                          <span>💬 {m.num_conversations} convos</span>
                        )}
                      </div>
                    </div>

                    {/* Right: Actions */}
                    <div className="flex items-center gap-2 shrink-0">
                      {!isActive && (
                        <button
                          onClick={() => handleActivate(m.id)}
                          disabled={isActing}
                          className="text-xs font-semibold px-3 py-1.5 bg-primary-500 hover:bg-primary-600 text-white rounded-lg transition-colors disabled:opacity-50"
                        >
                          {isActing ? "..." : "⚡ Activate"}
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(m.id)}
                        disabled={isActing}
                        className="text-xs font-semibold px-3 py-1.5 bg-red-50 hover:bg-red-100 text-red-600 rounded-lg transition-colors disabled:opacity-50"
                      >
                        {isActing ? "..." : "🗑️"}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ═══ PIPELINE INFO ════════════════════════════ */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            icon: "🤖", title: "BERT Detection",
            desc: "Pre-trained toxic-bert classifies each message across 6 toxicity categories.",
            badge: "Pre-trained", badgeColor: "bg-blue-100 text-blue-700",
          },
          {
            icon: "🌲", title: "Random Forest",
            desc: "Trained on 9 conversation features: toxicity trend, sentiment, bully ratio, etc.",
            badge: "Trainable", badgeColor: "bg-green-100 text-green-700",
          },
          {
            icon: "🧠", title: "LSTM (Deep Learning)",
            desc: "Learns temporal escalation patterns from sequences of toxicity scores over a conversation.",
            badge: "Deep Learning", badgeColor: "bg-purple-100 text-purple-700",
          },
        ].map((m) => (
          <div key={m.title} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
            <div className="text-3xl mb-3">{m.icon}</div>
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-semibold text-gray-800">{m.title}</h3>
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${m.badgeColor}`}>
                {m.badge}
              </span>
            </div>
            <p className="text-sm text-gray-500">{m.desc}</p>
          </div>
        ))}
      </div>

      {/* ═══ TRAIN NEW MODEL ══════════════════════════ */}
      <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">🚀 Train New Model</h2>

        {/* Required Columns Info */}
        <div className="mb-5 bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
          <p className="font-semibold mb-1">📋 Required CSV columns:</p>
          <code className="text-xs">id, conversation_id, user_id, timestamp, message, label</code>
          <p className="mt-2 text-xs text-blue-600">
            💡 Use <strong>formatted_train.csv</strong> from the <code>Datasets/</code> folder — it's already formatted correctly.
          </p>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`cursor-pointer border-2 border-dashed rounded-xl p-10 text-center transition-all ${
            isDragging
              ? "border-primary-500 bg-primary-50"
              : file
              ? "border-green-400 bg-green-50"
              : "border-gray-300 hover:border-primary-400 hover:bg-gray-50"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {file ? (
            <div>
              <div className="text-4xl mb-2">✅</div>
              <p className="font-semibold text-green-700">{file.name}</p>
              <p className="text-sm text-green-600 mt-1">
                {(file.size / 1024 / 1024).toFixed(2)} MB — Ready to train
              </p>
              <button
                className="mt-3 text-xs text-gray-400 hover:text-gray-600 underline"
                onClick={(e) => { e.stopPropagation(); setFile(null); setStatus("idle"); setResult(null); }}
              >
                Remove file
              </button>
            </div>
          ) : (
            <div>
              <div className="text-4xl mb-2">📂</div>
              <p className="text-gray-600 font-medium">Drag & drop your CSV here</p>
              <p className="text-sm text-gray-400 mt-1">or click to browse</p>
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
        <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm space-y-6 animate-fade-in">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-800">✅ Training Complete</h2>
            {result.model_id && (
              <span className="text-xs font-mono bg-gray-100 text-gray-500 px-3 py-1 rounded-full">
                ID: {result.model_id}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* BERT */}
            <div className="border border-blue-200 bg-blue-50 rounded-xl p-4">
              <div className="text-2xl mb-1">🤖</div>
              <h3 className="font-semibold text-blue-800 mb-2">BERT Detection</h3>
              <div className="text-sm text-blue-700 space-y-1">
                <p>Accuracy: <strong>{formatAccuracy(result.detection?.accuracy)}</strong></p>
                <p>Train size: <strong>{result.detection?.train_size ?? "N/A"}</strong></p>
                <p className="text-xs text-blue-500 mt-1">
                  {result.detection?.report?.status || "Pre-trained model active"}
                </p>
              </div>
            </div>

            {/* Random Forest */}
            <div className="border border-green-200 bg-green-50 rounded-xl p-4">
              <div className="text-2xl mb-1">🌲</div>
              <h3 className="font-semibold text-green-800 mb-2">Random Forest</h3>
              {result.escalation?.report ? (
                <div className="text-sm text-green-700 space-y-1">
                  {["LOW", "MEDIUM", "HIGH"].map((cls) => (
                    result.escalation.report[cls] && (
                      <p key={cls}>
                        {cls}: <strong>{formatAccuracy(result.escalation.report[cls]["f1-score"])}</strong> F1
                      </p>
                    )
                  ))}
                  <p className="text-xs text-green-500 mt-1">
                    Classes: {result.escalation?.classes?.join(", ")}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-green-600">{result.escalation?.error || "No data"}</p>
              )}
            </div>

            {/* LSTM */}
            <div className="border border-purple-200 bg-purple-50 rounded-xl p-4">
              <div className="text-2xl mb-1">🧠</div>
              <h3 className="font-semibold text-purple-800 mb-2">LSTM Escalation</h3>
              {result.lstm?.training_accuracy !== undefined ? (
                <div className="text-sm text-purple-700 space-y-1">
                  <p>Accuracy: <strong>{formatAccuracy(result.lstm.training_accuracy)}</strong></p>
                  <p>Conversations: <strong>{result.lstm.num_conversations}</strong></p>
                  <p>Epochs: <strong>{result.lstm.epochs}</strong></p>
                  <p>Final Loss: <strong>{result.lstm.final_loss}</strong></p>
                </div>
              ) : (
                <p className="text-sm text-purple-600">{result.lstm?.error || "No data"}</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
