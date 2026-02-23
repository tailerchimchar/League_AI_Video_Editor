import { useState, useCallback, useRef, useEffect } from "react";
import Markdown from "react-markdown";

interface Props {
  videoId: string;
}

export default function ReportView({ videoId }: Props) {
  const [analysis, setAnalysis] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const closeStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => closeStream();
  }, [closeStream]);

  const generateReport = useCallback(() => {
    closeStream();
    setAnalysis("");
    setError(null);
    setStreaming(true);

    const es = new EventSource(`/api/v1/videos/${videoId}/report`);
    esRef.current = es;

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "text") {
        setAnalysis((prev) => prev + data.text);
      } else if (data.type === "done") {
        setStreaming(false);
        es.close();
        esRef.current = null;
      } else if (data.type === "error") {
        setError(data.message);
        setStreaming(false);
        es.close();
        esRef.current = null;
      }
    };

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) return;
      setError("Lost connection to the report stream.");
      setStreaming(false);
      es.close();
      esRef.current = null;
    };
  }, [videoId, closeStream]);

  return (
    <div className="report-panel">
      <div className="report-header">
        <h2 className="panel-title">Evidence-Grounded Report</h2>
        <button
          className="report-btn"
          onClick={generateReport}
          disabled={streaming}
        >
          {streaming
            ? "Generating..."
            : analysis
              ? "Regenerate Report"
              : "Generate Report"}
        </button>
      </div>

      {error && (
        <div className="status-bar error">
          <span className="error-msg">{error}</span>
        </div>
      )}

      {(streaming || analysis) && (
        <div className="analysis-text">
          {analysis ? (
            <Markdown>{analysis}</Markdown>
          ) : (
            "Building evidence payload and generating report..."
          )}
        </div>
      )}
    </div>
  );
}
