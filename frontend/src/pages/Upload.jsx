import { useState } from "react";
import { api } from "../api";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.trainModels(file);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Training Data</h2>
        <p className="text-gray-500 mb-8">
          Upload a CSV file containing message data to train the detection and escalation models.
          Required columns: <code className="bg-gray-100 px-2 py-1 rounded text-sm">id, conversation_id, message</code>.
          Optional: <code className="bg-gray-100 px-2 py-1 rounded text-sm">label (1=bullying)</code>
        </p>

        <div className="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center hover:border-primary-400 transition-colors bg-gray-50 mb-6 relative">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="text-4xl mb-4">📄</div>
          {file ? (
            <div>
              <p className="font-semibold text-gray-800">{file.name}</p>
              <p className="text-gray-500 text-sm mt-1">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
          ) : (
            <div>
              <p className="font-semibold text-gray-700">Click or drag CSV file here</p>
              <p className="text-gray-500 text-sm mt-1">Maximum file size 50MB</p>
            </div>
          )}
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className={`w-full py-3 rounded-lg font-semibold text-white transition-all shadow-sm ${
            !file || loading
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-primary-600 hover:bg-primary-700 hover:shadow-md"
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Training Models...
            </span>
          ) : (
            "Start Training"
          )}
        </button>

        {error && (
          <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg border border-red-100 animate-slide-up">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {result && (
        <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 animate-slide-up space-y-6">
          <div className="flex items-center gap-3 text-success font-bold text-xl">
            <span>✅</span> Training Complete
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-5 rounded-xl border border-gray-100">
              <h3 className="font-semibold text-gray-800 mb-3 border-b pb-2">Detection Model</h3>
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex justify-between"><span>Accuracy:</span> <span className="font-medium text-gray-900">{(result.detection.accuracy * 100).toFixed(1)}%</span></li>
                <li className="flex justify-between"><span>Train Size:</span> <span className="font-medium text-gray-900">{result.detection.train_size}</span></li>
                <li className="flex justify-between"><span>Test Size:</span> <span className="font-medium text-gray-900">{result.detection.test_size}</span></li>
              </ul>
            </div>
            
            <div className="bg-gray-50 p-5 rounded-xl border border-gray-100">
              <h3 className="font-semibold text-gray-800 mb-3 border-b pb-2">Escalation Model</h3>
              {result.escalation?.classes ? (
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex justify-between"><span>Model:</span> <span className="font-medium text-gray-900">Random Forest</span></li>
                  <li className="flex justify-between"><span>Classes:</span> <span className="font-medium text-gray-900">{result.escalation.classes.join(", ")}</span></li>
                </ul>
              ) : (
                <p className="text-sm text-gray-500 italic mt-2">Using rule-based escalation (no conversation_id found in data or insufficient rows).</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
