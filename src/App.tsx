import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

interface TranslationResult {
  original_text: string;
  translated_text: string;
  status: string;
}

interface Selection {
  x: number;
  y: number;
  width: number;
  height: number;
}

function App() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null);
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Listen for global shortcut event from Rust
    const unlisten = listen("start-capture", () => {
      setIsCapturing(true);
      setResult(null);
    });
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isCapturing) return;
    setStartPos({ x: e.clientX, y: e.clientY });
    setSelection({ x: e.clientX, y: e.clientY, width: 0, height: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isCapturing || !startPos) return;

    const currentX = e.clientX;
    const currentY = e.clientY;

    const x = Math.min(startPos.x, currentX);
    const y = Math.min(startPos.y, currentY);
    const width = Math.abs(startPos.x - currentX);
    const height = Math.abs(startPos.y - currentY);

    setSelection({ x, y, width, height });
  };

  const handleMouseUp = async () => {
    if (!isCapturing || !selection || selection.width < 5) {
      setStartPos(null);
      setSelection(null);
      return;
    }

    setLoading(true);
    setIsCapturing(false);

    try {
      // Get DPI scaling factor
      const dpr = window.devicePixelRatio || 1;

      // Send coordinates to Rust (multiplied by DPR for physical pixels)
      const res = await invoke<TranslationResult>("process_screen_area", {
        x: Math.round(selection.x * dpr),
        y: Math.round(selection.y * dpr),
        width: Math.round(selection.width * dpr),
        height: Math.round(selection.height * dpr),
      });
      setResult(res);
    } catch (err) {
      console.error("OCR Error:", err);
    } finally {
      setLoading(false);
      setStartPos(null);
      setSelection(null);
      // Optional: hide window after some time or keep it to show results
      // await invoke("hide_window");
    }
  };

  return (
    <div 
      className={`app-container ${isCapturing ? "capturing" : ""}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      ref={containerRef}
    >
      {!isCapturing && !result && !loading && (
        <div className="welcome">
          <h1>OCR Translator</h1>
          <p>Press <strong>Ctrl + Shift + S</strong> to start capture</p>
          <button onClick={() => setIsCapturing(true)}>Start Manual Selection</button>
        </div>
      )}

      {isCapturing && (
        <div className="overlay">
          <div className="instruction">Drag to select area to translate</div>
          {selection && (
            <div 
              className="selection-box"
              style={{
                left: selection.x,
                top: selection.y,
                width: selection.width,
                height: selection.height
              }}
            />
          )}
        </div>
      )}

      {(loading || result) && (
        <div className="result-panel">
          {loading ? (
            <div className="loader">Processing...</div>
          ) : (
            result && (
              <div className="result-content">
                <div className="close-btn" onClick={() => setResult(null)}>×</div>
                <p className="original">{result.original_text}</p>
                <p className="translated">{result.translated_text}</p>
                <button className="hide-btn" onClick={() => invoke("hide_window")}>Hide Window</button>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}

export default App;
