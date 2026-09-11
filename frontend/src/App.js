import React, { useEffect, useState } from "react";
import "./App.css";

import InteractiveImageInvestigator from "./components/InteractiveImageInvestigator";
import InteractiveAudioInvestigator from "./components/InteractiveAudioInvestigator";
import InteractiveTextInvestigator from "./components/InteractiveTextInvestigator";
import PendingVerification from "./components/PendingVerification";

import {
  FaShieldAlt,
  FaFileAlt,
  FaImage,
  FaMicrophone,
  FaChartBar,
  FaUpload,
  FaBolt,
  FaBrain,
  FaDownload,
  FaExclamationTriangle,
  FaLock,
  FaSatelliteDish,
  FaFingerprint,
  FaDatabase,
  FaServer,
  FaCrosshairs,
  FaLayerGroup,
  FaNetworkWired,
  FaClock,
} from "react-icons/fa";

const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

function App() {
  const [page, setPage] = useState("dashboard");
  const [analyticsTab, setAnalyticsTab] = useState("overview");

  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);

  const [imagePreview, setImagePreview] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);

  const [result, setResult] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);

  const fetchPendingCount = async () => {
    try {
      const res = await fetch(`${API}/pending/count`);
      const data = await res.json();
      setPendingCount(data.count || 0);
    } catch (err) {
      console.error("Error fetching pending count:", err);
    }
  };

  useEffect(() => {
    fetchPendingCount();
  }, []);

  /*
  |--------------------------------------------------------------------------
  | Preview cleanup
  |--------------------------------------------------------------------------
  */

  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview);
      }
      if (audioPreview) {
        URL.revokeObjectURL(audioPreview);
      }
    };
  }, [imagePreview, audioPreview]);

  function clearImagePreview() {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setImagePreview(null);
  }

  function clearAudioPreview() {
    if (audioPreview) {
      URL.revokeObjectURL(audioPreview);
    }
    setAudioPreview(null);
  }

  function resetInputs() {
    clearImagePreview();
    clearAudioPreview();

    setText("");
    setFile(null);
    setAudioFile(null);
    setResult(null);
  }

  function openPage(nextPage) {
    resetInputs();
    setPage(nextPage);
    if (nextPage === "pending") {
      fetchPendingCount();
    }
  }

  /*
  |--------------------------------------------------------------------------
  | Generic helpers
  |--------------------------------------------------------------------------
  */

  function buildApiUrl(path) {
    if (!path) return null;
    const value = String(path);

    if (
      value.startsWith("http://") ||
      value.startsWith("https://") ||
      value.startsWith("blob:") ||
      value.startsWith("data:")
    ) {
      return value;
    }

    if (value.startsWith("/")) {
      return `${API}${value}`;
    }

    return `${API}/${value}`;
  }

  function normalizeProbability(value) {
    let number = Number(value || 0);
    if (number >= 0 && number <= 1) {
      number *= 100;
    }
    return Math.min(100, Math.max(0, number));
  }

  function riskClass(risk) {
    const value = String(risk || "low").toLowerCase();
    if (value.includes("high")) return "high";
    if (value.includes("medium")) return "medium";
    return "low";
  }

  function formatKey(key) {
    return String(key).replaceAll("_", " ").toUpperCase();
  }

  /*
  |--------------------------------------------------------------------------
  | VERIFY LATER HANDLER
  |--------------------------------------------------------------------------
  */

  async function handleVerifyLater() {
    if (!result) return;

    try {
      let identifier = text;
      let filepath = null;

      if (page === "image" && file) {
        identifier = file.name;
        filepath = result.uploaded_file || null;
      } else if (page === "audio" && audioFile) {
        identifier = audioFile.name;
        filepath = result.uploaded_file || null;
      } else if (page === "text" && file) {
        identifier = file.name;
      }

      const response = await fetch(`${API}/verify-later`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modality: page,
          identifier: identifier,
          prediction: result.prediction || "Fake",
          confidence: result.confidence || 0.0,
          filepath: filepath,
        }),
      });

      const data = await response.json();
      if (response.ok && data.status === "success") {
        alert("Result postponed to Pending Verification workspace.");
        fetchPendingCount();
      } else {
        alert("Failed to save for pending verification: " + (data.message || "Unknown error"));
      }
    } catch (err) {
      console.error("Error saving for later verification:", err);
      alert("Unable to reach backend API.");
    }
  }
  /*
|--------------------------------------------------------------------------
| CONTINUOUS LEARNING FEEDBACK HANDLERS
|--------------------------------------------------------------------------
*/
async function submitFeedback(verifiedLabel) {
  if (!result) return;

  // 1. Determine modality safely even when submitting from Dashboard
  let currentModality = page;
  if (page === "dashboard" || page === "analytics") {
    currentModality = String(
      result.modality || result.file_type || "text"
    ).toLowerCase();
  }

  // Ensure clean route strings (image, audio, text)
  if (currentModality.includes("image") || currentModality.includes("jpeg") || currentModality.includes("png")) {
    currentModality = "image";
  } else if (currentModality.includes("audio") || currentModality.includes("wav")) {
    currentModality = "audio";
  } else {
    currentModality = "text";
  }

  const endpoint = `${API}/feedback/${currentModality}`;

  let payload = {};

  if (currentModality === "image") {
    payload = {
        image_id: result.image_id, // Dynamically set by append_replay_record() from /analyze
        image_path: result.uploaded_file || result.image_path ||"",
        predicted_label: result.prediction || "AI",
        verified_label: verifiedLabel,
        confidence: parseFloat(result.confidence || 0),
        user_feedback: (result.prediction === verifiedLabel) ? "Agree" : "Disagree"
    };

    console.log("IMAGE FEEDBACK PAYLOAD:", payload);
}
   else if (currentModality === "text") {
    // 1. Extract sentence/text from backend result if file was uploaded
    let extractedText = "";
    
    if (result?.full_document && Array.isArray(result.full_document)) {
      extractedText = result.full_document.map(s => typeof s === "string" ? s : s.sentence || s.text).join(" ");
    } else if (typeof result?.full_document === "string") {
      extractedText = result.full_document;
    } else if (result?.text) {
      extractedText = result.text;
    }

    // 2. Fall back to raw text input, and only use file.name as absolute last resort
    const finalSentence = extractedText.trim() || text.trim() || file?.name || "Direct Text Entry";

    payload = {
      sentence: finalSentence,
      predicted_label: result?.prediction || "Fake",
      confidence: parseFloat(result?.confidence || 0.0),
      verified_label: verifiedLabel
    };
  
    } else if (currentModality === "audio") {
    payload = {
      audio_path: result.uploaded_file || "",
      audio_name: audioFile?.name || "sample.wav",
      predicted_label: result.prediction || "Fake",
      verified_label: verifiedLabel,
      confidence: result.confidence || 0.0
    };
  }

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (response.ok && data.status === "success") {
      alert(`Feedback saved! Model logged verification as: ${verifiedLabel}`);
    } else {
      alert("Failed to save feedback: " + (data.message || data.detail || `Status ${response.status}`));
    }
  } catch (err) {
    console.error("Error submitting feedback:", err);
    alert("Unable to reach backend feedback API.");
  }
}
  /*
  |--------------------------------------------------------------------------
  | API calls
  |--------------------------------------------------------------------------
  */

  async function submitAnalysis(type) {
    setLoading(true);
    setResult(null);

    const formData = new FormData();

    if (type === "text") {
      if (file) {
        formData.append("file", file);
      } else if (text.trim()) {
        formData.append("text", text.trim());
      }
    }

    if (type === "image" && file) {
      formData.append("file", file);
    }

    if (type === "audio" && audioFile) {
      formData.append("file", audioFile);
    }

    try {
      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        body: formData,
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error("The backend returned an invalid response.");
      }

      if (!response.ok) {
        throw new Error(
          data?.error || `Analysis failed with status ${response.status}`
        );
      }

      setResult(data);
    } catch (error) {
      setResult({
        error: error?.message || "Unable to connect to the FORGE backend.",
      });
    } finally {
      setLoading(false);
    }
  }

  async function loadAnalytics(tab = "overview") {
    setPage("analytics");
    setAnalyticsTab(tab);
    setResult(null);

    try {
      const response = await fetch(`${API}/analytics`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.error || `Analytics failed with status ${response.status}`
        );
      }

      setAnalytics(data);
    } catch (error) {
      setAnalytics({
        error: error?.message || "Unable to load analytics.",
      });
    }
  }

  /*
  |--------------------------------------------------------------------------
  | Layout components
  |--------------------------------------------------------------------------
  */

  const Sidebar = () => (
    <aside className="forge-sidebar">
      <div className="forge-brand">
        <div className="forge-logo">
          <FaShieldAlt />
          <span />
        </div>
        <div>
          <h2>F.O.R.G.E.</h2>
          <p>Forensic Observation & Recognition Gateway</p>
        </div>
      </div>

      <nav className="forge-nav">
        <button
          className={page === "dashboard" ? "active" : ""}
          onClick={() => openPage("dashboard")}
        >
          <FaServer />
          <span>Command Center</span>
        </button>

        <button
          className={page === "text" ? "active" : ""}
          onClick={() => openPage("text")}
        >
          <FaFileAlt />
          <span>Text Forensics</span>
        </button>

        <button
          className={page === "image" ? "active" : ""}
          onClick={() => openPage("image")}
        >
          <FaImage />
          <span>Image Forensics</span>
        </button>

        <button
          className={page === "audio" ? "active" : ""}
          onClick={() => openPage("audio")}
        >
          <FaMicrophone />
          <span>Audio Forensics</span>
        </button>

        <button
          className={page === "analytics" ? "active" : ""}
          onClick={() => loadAnalytics("overview")}
        >
          <FaChartBar />
          <span>Analytics Grid</span>
        </button>

        {/* Pending Verification Sidebar Item */}
        <button
          className={page === "pending" ? "active" : ""}
          onClick={() => openPage("pending")}
        >
          <FaClock />
          <span>
            Pending Verification{" "}
            {pendingCount > 0 && <span className="badge">{pendingCount}</span>}
          </span>
        </button>
      </nav>

      <div className="secure-panel">
        <p>Security Channel</p>
        <div>
          <span className="pulse-dot green" />
          Backend API Online
        </div>
        <div>
          <span className="pulse-dot green" />
          Report Generator Active
        </div>
        <div>
          <span className="pulse-dot cyan" />
          Multimodal XAI Enabled
        </div>
      </div>
    </aside>
  );

  const Header = ({ icon, title, subtitle }) => (
    <header className="forge-header page-snap">
      <div>
        <p className="forge-eyebrow">
          <FaLock />
          Secure AI Forensic Workstation
        </p>
        <h1>
          {icon}
          {title}
        </h1>
        <span>{subtitle}</span>
      </div>

      <div className="header-control-cluster">
        <div className="control-pill">
          <FaSatelliteDish />
          Local API
        </div>
        <div className="control-pill">
          <FaNetworkWired />
          Multimodal
        </div>
        <div className="control-pill hot">
          <FaCrosshairs />
          Live Triage
        </div>
      </div>
    </header>
  );

  const Loader = () => (
    <section className="scan-console">
      <div className="scan-cube">
        <FaFingerprint />
        <span />
      </div>
      <div>
        <h2>Running Forensic Scan</h2>
        <p>
          Extracting forensic features, executing detection models, generating
          explainable evidence, and producing the final risk profile.
        </p>
      </div>
      <div className="scan-bars">
        <i />
        <i />
        <i />
      </div>
    </section>
  );

  /*
  |--------------------------------------------------------------------------
  | Shared result components
  |--------------------------------------------------------------------------
  */

  const ResultPanel = () => {
    if (!result || result.error) return null;
    const confidence = normalizeProbability(result.confidence);

    return (
      <section className={`result-command ${riskClass(result.risk_level)}`}>
        <div className="result-holo">
          <div className="holo-ring">
            <svg viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="52" />
              <circle
                className="progress"
                cx="60"
                cy="60"
                r="52"
                style={{
                  strokeDashoffset: 327 - (327 * confidence) / 100,
                }}
              />
            </svg>
            <div>
              <strong>{confidence.toFixed(2)}%</strong>
              <span>confidence</span>
            </div>
          </div>
        </div>

        <div className="result-data">
          <p className="forge-eyebrow">Forensic Verdict</p>
          <h2>{result.prediction || "UNKNOWN"}</h2>

          <div className="result-metrics">
            <div>
              <span>Risk Level</span>
              <strong>{result.risk_level || "N/A"}</strong>
            </div>
            <div>
              <span>Risk Score</span>
              <strong>{result.risk_score ?? "N/A"}</strong>
            </div>
            <div>
              <span>Modality</span>
              <strong>
                {result.modality || result.file_type || "Analysis"}
              </strong>
            </div>
            <div>
              <span>Decision Strength</span>
              <strong>{result.decision_strength || "Computed"}</strong>
            </div>
          </div>

          {result.case_id && (
            <p className="result-note">
              Case ID: <b>{result.case_id}</b>
            </p>
          )}

          {result.recommendation && (
            <p className="result-note">{result.recommendation}</p>
          )}

          <div style={{ display: "flex", gap: "10px", marginTop: "15px" }}>
            {result.pdf_report && (
              <a
                className="forge-primary"
                href={buildApiUrl(result.pdf_report)}
                target="_blank"
                rel="noreferrer"
              >
                <FaDownload />
                Download FORGE Report
              </a>
            )}

            <button
              className="forge-secondary"
              onClick={handleVerifyLater}
              style={{ display: "flex", alignItems: "center", gap: "8px" }}
            >
              <FaClock />
              Verify Later
            </button>
          </div>

          {result.pdf_report_error && (
            <div className="forge-warning">
              <FaExclamationTriangle />
              Report error: {result.pdf_report_error}
            </div>
          )}
        </div>
      </section>
    );
  };

  const EvidenceMetadata = () => {
    if (!result?.evidence) return null;
    const evidence = result.evidence;

    return (
      <section className="forge-card">
        <div className="section-title">
          <FaFingerprint />
          <div>
            <h2>Evidence Integrity</h2>
            <p>
              File identity, hash and forensic examination metadata.
            </p>
          </div>
        </div>

        <div className="result-metrics">
          <div>
            <span>Filename</span>
            <strong>{evidence.original_filename || "N/A"}</strong>
          </div>
          <div>
            <span>MIME Type</span>
            <strong>{evidence.mime_type || "N/A"}</strong>
          </div>
          <div>
            <span>File Size</span>
            <strong>
              {evidence.size_bytes ? `${evidence.size_bytes} bytes` : "N/A"}
            </strong>
          </div>
          <div>
            <span>Analysis Version</span>
            <strong>
              {result.analysis_version ||
                result.audio_analysis_version ||
                result.image_analysis_version ||
                "FORGE"}
            </strong>
          </div>
        </div>

        {evidence.sha256 && (
          <p className="result-note">
            SHA-256: <code>{evidence.sha256}</code>
          </p>
        )}
      </section>
    );
  };

  const ProbabilityMatrix = () => {
    if (!result || result.error) return null;

    const ai = normalizeProbability(
      result?.probabilities?.ai ??
        result.raw_ai_probability ??
        result.raw_probability_fake ??
        result.risk_score ??
        0
    );

    const human = normalizeProbability(
      result?.probabilities?.human ??
        result.raw_human_probability ??
        result.raw_probability_real ??
        100 - ai
    );

    return (
      <section className="forge-card">
        <div className="section-title">
          <FaCrosshairs />
          <div>
            <h2>Probability Matrix</h2>
            <p>Synthetic and natural probability distribution.</p>
          </div>
        </div>

        <div className="probability-matrix">
          <div className="probability-row">
            <div>
              <span>AI / Fake Probability</span>
              <b>{ai.toFixed(2)}%</b>
            </div>
            <div className="prob-track">
              <i className="ai" style={{ width: `${ai}%` }} />
            </div>
          </div>

          <div className="probability-row">
            <div>
              <span>Human / Real Probability</span>
              <b>{human.toFixed(2)}%</b>
            </div>
            <div className="prob-track">
              <i className="human" style={{ width: `${human}%` }} />
            </div>
          </div>
        </div>
      </section>
    );
  };

  const ParameterGraph = () => {
    if (!result?.parameter_contribution) return null;

    return (
      <section className="forge-card">
        <div className="section-title">
          <FaLayerGroup />
          <div>
            <h2>Parameter Intelligence Grid</h2>
            <p>
              Forensic signal groups contributing to the final decision.
            </p>
          </div>
        </div>

        <div className="forge-graph">
          {Object.entries(result.parameter_contribution).map(
            ([key, value]) => {
              const score = normalizeProbability(
                typeof value === "object" ? value?.score : value
              );

              return (
                <div className="forge-graph-row" key={key}>
                  <div>
                    <span>{formatKey(key)}</span>
                    <b>{score.toFixed(2)}%</b>
                  </div>
                  <section>
                    <i
                      className={riskClass(value?.risk)}
                      style={{ width: `${score}%` }}
                    />
                  </section>
                </div>
              );
            }
          )}
        </div>
      </section>
    );
  };

  const ParameterCards = () => {
    if (!result?.parameter_contribution) return null;

    return (
      <section className="forge-card">
        <div className="section-title">
          <FaBrain />
          <div>
            <h2>XAI Reasoning Console</h2>
            <p>Human-readable explanations of the model decision.</p>
          </div>
        </div>

        <div className="xai-grid">
          {Object.entries(result.parameter_contribution).map(
            ([key, value]) => (
              <div
                className={`xai-card ${riskClass(value?.risk)}`}
                key={key}
              >
                <span className="xai-glow" />
                <div className="xai-head">
                  <h3>{formatKey(key)}</h3>
                  <b>{value?.risk || "LOW"}</b>
                </div>
                <strong>
                  {normalizeProbability(value?.score ?? value).toFixed(2)}%
                </strong>
                <p>{value?.reason || "No explanation was generated."}</p>
              </div>
            )
          )}
        </div>
      </section>
    );
  };

  const TextInvestigation = () => {
    if (!result || result.error) return null;
    return (
      <InteractiveTextInvestigator
        result={result}
        originalText={text}
        handleTextVerifyLater={handleVerifyLater}
      />
    );
  };

  const ImageInvestigation = () => {
    if (!result?.region_analysis || !result?.visual_evidence) return null;

    return (
      <InteractiveImageInvestigator
        originalUrl={imagePreview || buildApiUrl(result.uploaded_file)}
        visualEvidence={{
          heatmap: buildApiUrl(result.visual_evidence.heatmap),
          overlay: buildApiUrl(result.visual_evidence.overlay),
          naturalness_map: buildApiUrl(result.visual_evidence.naturalness_map),
          edge_map: buildApiUrl(result.visual_evidence.edge_map),
          frequency_map: buildApiUrl(result.visual_evidence.frequency_map),
          legend: result.visual_evidence.legend,
          interpretation_notice: result.visual_evidence.interpretation_notice,
        }}
        regionAnalysis={result.region_analysis}
      />
    );
  };

  const AudioInvestigation = () => {
    if (!result || result.error || !result.audio_timeline) return null;

    const source = audioPreview || buildApiUrl(result?.uploaded_file);
    if (!source) return null;

    const audioCurves =
      result?.audio_curves ||
      result?.advanced_audio_analysis?.curves || {
        pitch: [],
        energy: [],
        spectral_flux: [],
        spectral_flatness: [],
      };

    const voiceDNA =
      result?.voice_dna || result?.advanced_audio_analysis?.voice_dna || {};

    const audioSummary =
      result?.audio_summary ||
      result?.advanced_audio_analysis?.summary || {};

    const pauseIntervals =
      result?.pause_intervals ||
      result?.advanced_audio_analysis?.pause_intervals || [];

    const breathingEvents =
      result?.breathing_events ||
      result?.advanced_audio_analysis?.breathing_events || [];

    return (
      <InteractiveAudioInvestigator
        audioUrl={source}
        audioTimeline={result.audio_timeline}
        waveformUrl={buildApiUrl(result?.waveform)}
        spectrogramUrl={buildApiUrl(result?.spectrogram)}
        heatmapUrl={buildApiUrl(result?.audio_heatmap)}
        voiceDNA={voiceDNA}
        audioCurves={audioCurves}
        audioSummary={audioSummary}
        pauseIntervals={pauseIntervals}
        breathingEvents={breathingEvents}
      />
    );
  };
  const FeedbackPanel = () => {
  if (!result || result.error) return null;

  return (
    <section className="forge-card continuous-learning-panel" style={{ margin: "20px 0" }}>
      <div className="section-title">
        <FaBrain />
        <div>
          <h2>Continuous Learning Feedback</h2>
          <p>
            Verify model accuracy to update active learning datasets and trigger continuous retraining pipeline.
          </p>
        </div>
      </div>

      <div style={{ display: "flex", gap: "12px", marginTop: "16px" }}>
        <button
          type="button"
          onClick={() => submitFeedback("HUMAN")}
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "12px",
            background: "rgba(0, 230, 118, 0.15)",
            border: "1px solid #00e676",
            color: "#64f3aa",
            fontWeight: "bold",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px"
          }}
        >
          ✓ Mark as REAL (Human)
        </button>

        <button
          type="button"
          onClick={() => submitFeedback("AI")}
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "12px",
            background: "rgba(255, 61, 113, 0.15)",
            border: "1px solid #ff3d71",
            color: "#ff7f9f",
            fontWeight: "bold",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px"
          }}
        >
          ✕ Mark as FAKE (AI Generated)
        </button>

        <button
          type="button"
          onClick={handleVerifyLater}
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "12px",
            background: "rgba(255, 176, 0, 0.15)",
            border: "1px solid #ffb000",
            color: "#ffd166",
            fontWeight: "bold",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px"
          }}
        >
          <FaClock />
          Defer to Pending Workspace
        </button>
      </div>
    </section>
  );
};

  const ResultStack = ({ type }) => (
  <>
    {loading && <Loader />}

    {result?.error && (
      <section className="forge-error">
        <FaExclamationTriangle />
        {result.error}
      </section>
    )}

    {/* 1. TOP CARD: Forensic Verdict */}
    <ResultPanel />

    {/* 2. MIDDLE CARD: Continuous Learning Feedback */}
    <FeedbackPanel />

    {/* 3. BOTTOM CARD: Evidence Integrity */}
    <EvidenceMetadata />

    {type !== "text" && <ProbabilityMatrix />}
    {type === "text" && <TextInvestigation />}
    {type === "image" && <ImageInvestigation />}
    {type === "audio" && <AudioInvestigation />}

    {type !== "text" && (
      <>
        <ParameterGraph />
        <ParameterCards />
      </>
    )}
  </>
);

  /*
  |--------------------------------------------------------------------------
  | Views
  |--------------------------------------------------------------------------
  */

  const Dashboard = () => (
    <main className="forge-workspace page-snap">
      <Header
        icon={<FaShieldAlt />}
        title="FORGE Command Center"
        subtitle="Multimodal deepfake detection, explainable AI and digital forensic evidence analysis."
      />

      <section className="soc-hero">
        <div className="soc-left">
          <p className="forge-eyebrow">
            Multimodal Digital Forensics Console
          </p>
          <h2>
            Deepfake Threat Intelligence and Explainable Evidence Analysis
          </h2>
          <p>
            Analyse suspicious text, image and audio evidence using trained
            models, forensic feature extraction, visual explainability and
            downloadable reports.
          </p>

          <div className="soc-actions">
            <button className="forge-primary" onClick={() => openPage("text")}>
              <FaBolt />
              Begin Investigation
            </button>
            <button
              className="forge-secondary"
              onClick={() => loadAnalytics("overview")}
            >
              <FaChartBar />
              Open Analytics
            </button>
          </div>
        </div>

        <div className="forge-core">
          <div className="core-ring ring-one" />
          <div className="core-ring ring-two" />
          <div className="core-ring ring-three" />
          <div className="core-center">
            <FaShieldAlt />
            <span>FORGE</span>
          </div>
          <div className="core-scan-line" />
        </div>
      </section>

      <section className="threat-stats">
        <div className="threat-stat-card">
          <FaFileAlt />
          <span>Text Engine</span>
          <strong>ONLINE</strong>
          <p>Stylometry • TF-IDF • N-Gram • SBERT • SHAP</p>
        </div>

        <div className="threat-stat-card">
          <FaImage />
          <span>Image Engine</span>
          <strong>ONLINE</strong>
          <p>CNN • Random Forest • Region XAI • Heatmaps</p>
        </div>

        <div className="threat-stat-card">
          <FaMicrophone />
          <span>Audio Engine</span>
          <strong>ONLINE</strong>
          <p>LFCC • CNN-BiLSTM • Segment Timeline • Acoustic XAI</p>
        </div>

        <div className="threat-stat-card danger">
          <FaCrosshairs />
          <span>Investigation Mode</span>
          <strong>ACTIVE</strong>
          <p>Explainable evidence triage enabled</p>
        </div>
      </section>

      <section className="mission-grid">
        <div className="mission-card" onClick={() => openPage("text")}>
          <FaFileAlt />
          <h3>Text Forensics</h3>
          <p>Detect synthetic writing and inspect suspicious sentences.</p>
          <span>Launch Text Module</span>
        </div>

        <div className="mission-card" onClick={() => openPage("image")}>
          <FaImage />
          <h3>Image Forensics</h3>
          <p>Analyse generated images, regional abnormalities and heatmaps.</p>
          <span>Launch Image Module</span>
        </div>

        <div className="mission-card" onClick={() => openPage("audio")}>
          <FaMicrophone />
          <h3>Audio Forensics</h3>
          <p>Detect synthetic speech with timeline-level acoustic evidence.</p>
          <span>Launch Audio Module</span>
        </div>
      </section>
    </main>
  );

  const TextPage = () => (
    <main className="forge-workspace page-snap">
      <Header
        icon={<FaFileAlt />}
        title="Text Forensics"
        subtitle="Analyse raw text, DOCX, PDF and TXT evidence with sentence-level explainability."
      />

      <section className="analysis-grid">
        <div className="evidence-input">
          <h2>Evidence Intake</h2>

          <textarea
            placeholder="Paste suspicious text evidence here..."
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              if (file) setFile(null);
              setResult(null);
            }}
          />

          <label className="forge-upload">
            <FaUpload />
            Upload DOCX / PDF / TXT
            <input
              type="file"
              accept=".docx,.pdf,.txt"
              onChange={(event) => {
                const selectedFile = event.target.files?.[0] || null;
                setFile(selectedFile);
                setText("");
                setResult(null);
              }}
            />
          </label>

          {file && (
            <div className="file-chip">
              Selected Evidence: <b>{file.name}</b>
            </div>
          )}

          <button
            className="forge-primary full"
            disabled={loading || (!text.trim() && !file)}
            onClick={() => submitAnalysis("text")}
          >
            <FaBolt />
            Execute Text Scan
          </button>
        </div>

        <div className="module-intel">
          <h2>Text Signal Stack</h2>
          <p>Stylometric variance</p>
          <p>TF-IDF vocabulary fingerprint</p>
          <p>N-Gram phrase patterning</p>
          <p>SBERT semantic behaviour</p>
          <p>SHAP explanation layer</p>
        </div>
      </section>

      <ResultStack type="text" />
    </main>
  );

  const ImagePage = () => (
    <main className="forge-workspace page-snap">
      <Header
        icon={<FaImage />}
        title="Image Forensics"
        subtitle="Analyse generated images, regional abnormalities and visual evidence."
      />

      <section className="analysis-grid">
        <div className="evidence-input">
          <h2>Image Evidence Intake</h2>

          <label className="forge-upload large">
            <FaUpload />
            Upload PNG / JPG / JPEG / WEBP
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp"
              onChange={(event) => {
                const selectedFile = event.target.files?.[0] || null;
                clearImagePreview();
                setFile(selectedFile);
                setResult(null);

                if (selectedFile) {
                  setImagePreview(URL.createObjectURL(selectedFile));
                }
              }}
            />
          </label>

          {file && (
            <div className="file-chip">
              Selected Evidence: <b>{file.name}</b>
            </div>
          )}

          {imagePreview && (
            <img
              className="preview-frame"
              src={imagePreview}
              alt="Selected forensic evidence"
            />
          )}

          <button
            className="forge-primary full"
            disabled={loading || !file}
            onClick={() => submitAnalysis("image")}
          >
            <FaBolt />
            Execute Image Scan
          </button>
        </div>

        <div className="module-intel">
          <h2>Image Signal Stack</h2>
          <p>CNN visual inference</p>
          <p>Random Forest feature fusion</p>
          <p>Texture and noise analysis</p>
          <p>Patch-level hover investigation</p>
          <p>Heatmap and naturalness layers</p>
        </div>
      </section>

      <ResultStack type="image" />
    </main>
  );

  const AudioPage = () => (
    <main className="forge-workspace page-snap">
      <Header
        icon={<FaMicrophone />}
        title="Audio Forensics"
        subtitle="Analyse synthetic speech using LFCC, CNN-BiLSTM and timeline-level acoustic XAI."
      />

      <section className="analysis-grid">
        <div className="evidence-input">
          <h2>Audio Evidence Intake</h2>

          <label className="forge-upload large">
            <FaUpload />
            Upload WAV / FLAC / MP3 / M4A
            <input
              type="file"
              accept=".wav,.flac,.mp3,.m4a"
              onChange={(event) => {
                const selectedFile = event.target.files?.[0] || null;
                clearAudioPreview();
                setAudioFile(selectedFile);
                setResult(null);

                if (selectedFile) {
                  setAudioPreview(URL.createObjectURL(selectedFile));
                }
              }}
            />
          </label>

          {audioFile && (
            <div className="audio-chip">
              <p>{audioFile.name}</p>
              {audioPreview && <audio controls src={audioPreview} />}
            </div>
          )}

          <button
            className="forge-primary full"
            disabled={loading || !audioFile}
            onClick={() => submitAnalysis("audio")}
          >
            <FaBolt />
            Execute Audio Scan
          </button>
        </div>

        <div className="module-intel">
          <h2>Audio Signal Stack</h2>
          <p>LFCC spectral extraction</p>
          <p>CNN-BiLSTM fusion model</p>
          <p>Pitch, phase and energy analysis</p>
          <p>Segment-level risk timeline</p>
          <p>Click-to-seek acoustic investigation</p>
        </div>
      </section>

      <ResultStack type="audio" />
    </main>
  );

  const AnalyticsPage = () => {
    const data = analytics || {};
    const cards = Object.entries(data).filter(([key]) => key !== "error");

    const filteredCards = cards.filter(([key]) => {
      if (analyticsTab === "overview") return true;
      return key.toLowerCase().includes(analyticsTab);
    });

    return (
      <main className="forge-workspace page-snap">
        <Header
          icon={<FaChartBar />}
          title="Analytics Grid"
          subtitle="Operational statistics across FORGE forensic modules."
        />

        <section className="analytics-switcher">
          <button
            className={analyticsTab === "overview" ? "active" : ""}
            onClick={() => setAnalyticsTab("overview")}
          >
            <FaDatabase />
            Overview
          </button>

          <button
            className={analyticsTab === "text" ? "active" : ""}
            onClick={() => setAnalyticsTab("text")}
          >
            <FaFileAlt />
            Text
          </button>

          <button
            className={analyticsTab === "image" ? "active" : ""}
            onClick={() => setAnalyticsTab("image")}
          >
            <FaImage />
            Image
          </button>

          <button
            className={analyticsTab === "audio" ? "active" : ""}
            onClick={() => setAnalyticsTab("audio")}
          >
            <FaMicrophone />
            Audio
          </button>
        </section>

        {analytics?.error && (
          <section className="forge-error">
            <FaExclamationTriangle />
            {analytics.error}
          </section>
        )}

        {!analytics?.error && (
          <section className="analytics-grid">
            {filteredCards.map(([key, value]) => (
              <div className="analytics-card" key={key}>
                <span>{formatKey(key)}</span>
                <strong>
                  {typeof value === "object"
                    ? JSON.stringify(value)
                    : String(value)}
                </strong>
              </div>
            ))}
          </section>
        )}
      </main>
    );
  };

  const PendingPage = () => (
    <main className="forge-workspace page-snap">
      <Header
        icon={<FaClock />}
        title="Pending Verification Workspace"
        subtitle="Review postponed items before sending them to the continuous learning retraining pipeline."
      />
      <PendingVerification onCountUpdate={setPendingCount} />
    </main>
  );

  /*
  |--------------------------------------------------------------------------
  | Main render
  |--------------------------------------------------------------------------
  */

  return (
    <div className="forge-app">
      <div className="forge-bg-grid" />
      <div className="forge-noise" />

      <div className="orb orb-a" />
      <div className="orb orb-b" />
      <div className="orb orb-c" />

      <Sidebar />

      {page === "dashboard" && <Dashboard />}
      {page === "text" && <TextPage />}
      {page === "image" && <ImagePage />}
      {page === "audio" && <AudioPage />}
      {page === "analytics" && <AnalyticsPage />}
      {page === "pending" && <PendingPage />}
    </div>
  );
}

export default App;