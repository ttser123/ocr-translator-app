import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

interface TranslationResult {
  original_text: string;
  translated_text: string;
  status: string;
}

function App() {
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function processScreen() {
    setLoading(true);
    try {
      // Calling process_screen_area from main.rs
      const res = await invoke<TranslationResult>("process_screen_area", { 
        x: 0, 
        y: 0, 
        width: 100, 
        height: 100 
      });
      setResult(res);
    } catch (err) {
      console.error("Error calling process_screen_area:", err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>OCR Translator</h1>
      
      <div className="card">
        <button onClick={processScreen} disabled={loading}>
          {loading ? "Processing..." : "Process Screen (Warrior)"}
        </button>
      </div>

      {result && (
        <div className="result-container" style={{ marginTop: "20px", textAlign: "left" }}>
          <p><strong>Original:</strong> {result.original_text}</p>
          <p><strong>Translated:</strong> {result.translated_text}</p>
          <p><strong>Status:</strong> {result.status}</p>
        </div>
      )}
    </main>
  );
}

export default App;
