import { useEffect, useState } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { ImageGrid } from "./components/ImageGrid";
import { ConfidenceMeter } from "./components/ConfidenceMeter";
import { PassagesList } from "./components/PassagesList";
import { HealthBanner } from "./components/HealthBanner";
import { diagnose, fetchHealth, searchSimilarImages } from "./lib/api";
import type { DiagnoseResponse, HealthResponse, SearchResponse } from "./types";
import "./App.css";

type Tab = "search" | "diagnose";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("diagnose");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [diagnoseResult, setDiagnoseResult] = useState<DiagnoseResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message));
  }, []);

  async function run() {
    if (!file) return;
    setLoading(true);
    setRunError(null);
    try {
      if (tab === "search") {
        setSearchResult(await searchSimilarImages(file, 5));
      } else {
        setDiagnoseResult(await diagnose(file, 3));
      }
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>Visual Fault Diagnostics</h1>
        <p className="subtitle">
          Upload a part image to find visually similar historical faults and, with
          AI Diagnosis, pull the relevant maintenance-manual remediation steps. React/TypeScript
          frontend for{" "}
          <a
            href="https://github.com/manuelbomi/GPU-Accelerated-Visual-Fault-Diagnostics-with-RAG-Maintenance-PDF-Intelligence"
            target="_blank"
            rel="noreferrer"
          >
            GPU-Accelerated-Visual-Fault-Diagnostics-with-RAG-Maintenance-PDF-Intelligence
          </a>
          .
        </p>
      </header>

      <HealthBanner health={health} error={healthError} />

      <section className="panel">
        <ImageUploader onFileSelected={setFile} disabled={loading} />
      </section>

      <section className="panel">
        <div className="tabs" role="tablist">
          <button
            role="tab"
            aria-selected={tab === "diagnose"}
            className={tab === "diagnose" ? "tab active" : "tab"}
            onClick={() => setTab("diagnose")}
          >
            AI Diagnosis
          </button>
          <button
            role="tab"
            aria-selected={tab === "search"}
            className={tab === "search" ? "tab active" : "tab"}
            onClick={() => setTab("search")}
          >
            Visual Search
          </button>
        </div>

        <button className="run-button" onClick={run} disabled={!file || loading}>
          {loading ? "Running..." : tab === "diagnose" ? "Run AI diagnosis" : "Find similar images"}
        </button>

        {runError && <p className="error">{runError}</p>}

        {tab === "search" && searchResult && (
          <div className="results">
            <h3>Similar images to {searchResult.query_filename}</h3>
            <ImageGrid items={searchResult.results} />
          </div>
        )}

        {tab === "diagnose" && diagnoseResult && (
          <div className="results">
            {diagnoseResult.predicted_label && diagnoseResult.predicted_confidence !== null && (
              <ConfidenceMeter
                label={diagnoseResult.predicted_label}
                confidence={diagnoseResult.predicted_confidence}
              />
            )}

            <h3>Similar historical faults</h3>
            <ImageGrid items={diagnoseResult.top_images} />

            <h3>Relevant maintenance passages</h3>
            {diagnoseResult.related_docs.length > 0 && (
              <p className="muted">
                From: {diagnoseResult.related_docs.join(", ")}
              </p>
            )}
            <PassagesList passages={diagnoseResult.retrieved_passages} />

            <h3>AI-synthesized remediation</h3>
            {diagnoseResult.synthesis ? (
              <p className="synthesis">{diagnoseResult.synthesis}</p>
            ) : (
              <p className="muted">
                Not configured — set the <code>OPENAI_API_KEY</code> environment
                variable on the API server to enable LLM-synthesized remediation
                summaries.
              </p>
            )}
          </div>
        )}
      </section>

      <footer className="page-footer">
        <p>
          Backend: FastAPI wrapping <code>src/query_similar_images.py</code> and{" "}
          <code>src/rag_query.py</code>. See <code>api/README.md</code> and{" "}
          <code>frontend/README.md</code>.
        </p>
      </footer>
    </div>
  );
}
