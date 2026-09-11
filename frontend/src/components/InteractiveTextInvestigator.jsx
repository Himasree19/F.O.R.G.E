import React, { useMemo, useState } from "react";
import "./InteractiveTextInvestigator.css";

function clamp(value, min = 0, max = 100) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "0.0";
}

function normalizeRisk(value, score = 0) {
  const risk = String(value || "").toUpperCase();

  if (risk.includes("HIGH")) return "HIGH";
  if (risk.includes("MEDIUM") || risk.includes("MODERATE")) return "MEDIUM";
  if (risk.includes("LOW")) return "LOW";

  if (score >= 70) return "HIGH";
  if (score >= 40) return "MEDIUM";
  return "LOW";
}

function riskClass(risk, score) {
  return `risk-${normalizeRisk(risk, score).toLowerCase()}`;
}

function normalizeSentences(result) {
  const source =
    result?.full_document ||
    result?.highlighted_document ||
    result?.suspicious_sentences ||
    [];

  if (!Array.isArray(source)) {
    return [];
  }

  return source.map((item, index) => {
    if (typeof item === "string") {
      return {
        id: `sentence-${index}`,
        sentence: item,
        score: 0,
        risk: "LOW",
        reason: "No sentence-level explanation was returned.",
      };
    }

    const score = clamp(
      item?.score ??
      item?.ai_score ??
      item?.confidence ??
      item?.probability ??
      0
    );

    return {
      ...item,
      id: item?.id || `sentence-${index}`,
      sentence: item?.sentence || item?.text || "",
      score,
      risk: normalizeRisk(item?.risk || item?.risk_level, score),
      reason:
        item?.reason ||
        item?.explanation ||
        item?.interpretation ||
        "This sentence contains measurable linguistic patterns used by the text-forensic model.",
      indicators:
        item?.indicators ||
        item?.findings ||
        [],
    };
  });
}

function normalizeParameters(parameters) {
  return Object.entries(parameters || {}).map(([key, value]) => {
    const score =
      typeof value === "object"
        ? Number(value?.score || 0)
        : Number(value || 0);

    return {
      key,
      label: key.replaceAll("_", " "),
      score: clamp(score),
      risk:
        typeof value === "object"
          ? value?.risk || normalizeRisk("", score)
          : normalizeRisk("", score),
      reason:
        typeof value === "object"
          ? value?.reason || ""
          : "",
    };
  });
}

function SentencePanel({ sentence, locked, onUnlock, onVerifyLater }) {
  if (!sentence) {
    return (
      <aside className="text-live-console text-live-console-empty">
        <div className="text-console-radar">
          <span />
        </div>

        <p className="text-eyebrow">
          LIVE SENTENCE ANALYSIS
        </p>

        <h3>Select a sentence</h3>

        <p>
          Hover over a sentence for a preview, or click it to lock the
          analysis and inspect its linguistic indicators.
        </p>
      </aside>
    );
  }

  return (
    <aside className="text-live-console">
      <div className="text-console-heading">
        <div>
          <p className="text-eyebrow">
            {locked
              ? "LOCKED SENTENCE ANALYSIS"
              : "LIVE SENTENCE ANALYSIS"}
          </p>
          <h3>Sentence Evidence</h3>
        </div>

        <span className={riskClass(sentence.risk, sentence.score)}>
          {sentence.risk}
        </span>
      </div>

      <div className="text-score-grid">
        <div
          className="text-score-ring"
          style={{
            "--score": `${sentence.score * 3.6}deg`,
          }}
        >
          <div>
            <strong>{formatNumber(sentence.score)}%</strong>
            <span>AI suspicion</span>
          </div>
        </div>

        <div className="text-score-card">
          <span>Sentence probability</span>
          <strong>{formatNumber(sentence.score)}%</strong>
          <div>
            <i style={{ width: `${sentence.score}%` }} />
          </div>
        </div>
      </div>

      <section className="text-console-section">
        <h4>Selected sentence</h4>
        <blockquote>{sentence.sentence}</blockquote>
      </section>

      <section className="text-console-section">
        <h4>AI interpretation</h4>
        <p>{sentence.reason}</p>
      </section>

      {Array.isArray(sentence.indicators) &&
        sentence.indicators.length > 0 && (
          <section className="text-console-section">
            <h4>Detected indicators</h4>
            <ul className="text-findings-list">
              {sentence.indicators.map((indicator) => (
                <li key={indicator}>{indicator}</li>
              ))}
            </ul>
          </section>
        )}

      <section className="text-console-section text-console-recommendation">
        <h4>Recommended review</h4>
        <p>
          Compare this sentence with neighbouring sentences and the global
          stylometric, lexical and semantic model outputs.
        </p>
      </section>

      {locked && (
        <button
          type="button"
          className="text-unlock-button"
          onClick={onUnlock}
        >
          Unlock sentence
        </button>
      )}
      {/* FEEDBACK & VERIFY LATER BUTTONS */}
      <section className="text-console-section" style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
        <h4>Continuous Learning Feedback</h4>
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
          <button
            type="button"
            onClick={() => alert("Flagged as Correct!")}
            style={{ flex: 1, padding: "0.5rem", borderRadius: "6px", background: "#00e676", color: "#000", border: "none", fontWeight: "bold", cursor: "pointer" }}
          >
            ✓ HUMAN
          </button>
          <button
            type="button"
            onClick={() => alert("Flagged as Incorrect!")}
            style={{ flex: 1, padding: "0.5rem", borderRadius: "6px", background: "#ff3d71", color: "#fff", border: "none", fontWeight: "bold", cursor: "pointer" }}
          >
            ✕ AI
          </button>
          <button
            type="button"
            onClick={onVerifyLater}
            style={{ flex: 1, padding: "0.5rem", borderRadius: "6px", background: "#ffd166", color: "#000", border: "none", fontWeight: "bold", cursor: "pointer" }}
          >
            ⏱ Verify Later
        </button>
        </div>
      </section>
    </aside>
  );
}

function ParameterGrid({ parameters }) {
  if (!parameters.length) {
    return (
      <div className="text-empty-state">
        No parameter-level XAI evidence was returned.
      </div>
    );
  }

  return (
    <div className="text-parameter-grid">
      {parameters.map((parameter) => (
        <article
          className="text-parameter-card"
          key={parameter.key}
        >
          <div>
            <strong>{parameter.label}</strong>
            <span>{formatNumber(parameter.score)}%</span>
          </div>

          <section>
            <i style={{ width: `${parameter.score}%` }} />
          </section>

          {parameter.reason && (
            <p>{parameter.reason}</p>
          )}
        </article>
      ))}
    </div>
  );
}

function SentenceMap({ sentences, selectedId, onSelect }) {
  if (!sentences.length) {
    return (
      <div className="text-empty-state">
        No sentence-level evidence was returned.
      </div>
    );
  }

  return (
    <div className="text-sentence-map">
      {sentences.map((sentence, index) => (
        <button
          type="button"
          className={[
            "text-sentence-map-card",
            riskClass(sentence.risk, sentence.score),
            selectedId === sentence.id ? "active" : "",
          ].join(" ")}
          key={sentence.id}
          onClick={() => onSelect(sentence)}
        >
          <span>{String(index + 1).padStart(2, "0")}</span>

          <div>
            <strong>
              {sentence.sentence.slice(0, 80)}
              {sentence.sentence.length > 80 ? "…" : ""}
            </strong>

            <section>
              <i style={{ width: `${sentence.score}%` }} />
            </section>
          </div>

          <b>{formatNumber(sentence.score)}%</b>
        </button>
      ))}
    </div>
  );
}

function InteractiveTextInvestigator({ result = {}, originalText = "" }) {
  const [activeView, setActiveView] = useState("document");
  const [hoveredSentence, setHoveredSentence] = useState(null);
  const [selectedSentence, setSelectedSentence] = useState(null);

  const sentences = useMemo(
    () => normalizeSentences(result),
    [result]
  );

  const parameters = useMemo(
    () => normalizeParameters(result?.parameter_contribution),
    [result]
  );

  const displayedSentence =
    hoveredSentence ||
    selectedSentence;

  const locked =
    Boolean(
      selectedSentence &&
      !hoveredSentence
    );

  const fakeProbability = clamp(
    result?.raw_probability_fake ??
    result?.fake_probability ??
    (String(result?.prediction || "").toUpperCase().includes("AI")
      ? result?.confidence
      : 100 - Number(result?.confidence || 0))
  );

  const realProbability = clamp(
    result?.raw_probability_real ??
    result?.real_probability ??
    100 - fakeProbability
  );
async function handleTextVerifyLater(e) {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:8000/verify-later", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modality: "text",
          identifier: originalText || result?.text || "Text_Sample",
          prediction: result?.prediction || "Fake",
          confidence: fakeProbability / 100 || 0.85,
        }),
      });

      if (response.ok) {
        alert("Text sample moved to Pending Verification!");
      } else {
        alert("Failed to move text to pending verification.");
      }
    } catch (err) {
      console.error("Error staging text for verification:", err);
    }
  }
  return (
    <section className="text-investigator">
      <header className="text-investigator-header">
        <div>
          <p className="text-eyebrow">
            FORGE TEXT XAI 3.0
          </p>

          <h2>
            Interactive Text Forensic Workstation
          </h2>

          <p>
            Inspect sentence-level evidence, compare linguistic signals and
            lock suspicious passages for detailed review.
          </p>
        </div>

        <div className="text-investigator-status">
          <span>{sentences.length} analysed sentences</span>
          <span>{result?.prediction || "Unknown verdict"}</span>
          <span>{result?.risk_level || "Risk unavailable"}</span>
        </div>
      </header>

      <nav className="text-view-tabs">
        <div>
          {[
            ["document", "Document"],
            ["sentence-map", "Sentence Map"],
            ["parameters", "XAI Parameters"],
          ].map(([key, label]) => (
            <button
              type="button"
              className={activeView === key ? "active" : ""}
              key={key}
              onClick={() => setActiveView(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </nav>

      <section className="text-probability-panel">
        <div className="text-section-heading">
          <div>
            <p className="text-eyebrow">PROBABILITY MATRIX</p>
            <h3>Synthetic and natural probability distribution</h3>
          </div>
        </div>

        <div className="text-probability-row fake">
          <div>
            <span>AI / Fake probability</span>
            <strong>{formatNumber(fakeProbability)}%</strong>
          </div>
          <section>
            <i style={{ width: `${fakeProbability}%` }} />
          </section>
        </div>

        <div className="text-probability-row real">
          <div>
            <span>Human / Real probability</span>
            <strong>{formatNumber(realProbability)}%</strong>
          </div>
          <section>
            <i style={{ width: `${realProbability}%` }} />
          </section>
        </div>
      </section>

      <div className="text-workstation-grid">
        <div className="text-evidence-column">
          {activeView === "document" && (
            <section className="text-document-panel">
              <div className="text-section-heading">
                <div>
                  <p className="text-eyebrow">
                    SENTENCE EVIDENCE
                  </p>
                  <h3>
                    Interactive document examination
                  </h3>
                </div>

                <span>
                  Hover to preview • Click to lock
                </span>
              </div>

              <div className="text-document-surface">
                {sentences.length ? (
                  sentences.map((sentence, index) => (
                    <button
                      type="button"
                      className={[
                        "text-sentence-chip",
                        riskClass(sentence.risk, sentence.score),
                        selectedSentence?.id === sentence.id
                          ? "selected"
                          : "",
                        hoveredSentence?.id === sentence.id
                          ? "hovered"
                          : "",
                      ].join(" ")}
                      key={sentence.id}
                      onMouseEnter={() =>
                        setHoveredSentence(sentence)
                      }
                      onMouseLeave={() =>
                        setHoveredSentence(null)
                      }
                      onClick={() =>
                        setSelectedSentence(sentence)
                      }
                    >
                      <span className="text-sentence-number">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <span>{sentence.sentence}</span>
                      <b>{formatNumber(sentence.score)}%</b>
                    </button>
                  ))
                ) : (
                  <p className="text-document-raw">
                    {originalText ||
                      result?.text ||
                      "No document content was returned."}
                  </p>
                )}
              </div>
            </section>
          )}

          {activeView === "sentence-map" && (
            <section className="text-document-panel">
              <div className="text-section-heading">
                <div>
                  <p className="text-eyebrow">
                    SENTENCE RISK MAP
                  </p>
                  <h3>
                    Ranked sentence-level evidence
                  </h3>
                </div>
              </div>

              <SentenceMap
                sentences={sentences}
                selectedId={selectedSentence?.id}
                onSelect={(sentence) => {
                  setSelectedSentence(sentence);
                  setHoveredSentence(null);
                }}
              />
            </section>
          )}

          {activeView === "parameters" && (
            <section className="text-document-panel">
              <div className="text-section-heading">
                <div>
                  <p className="text-eyebrow">
                    PARAMETER INTELLIGENCE
                  </p>
                  <h3>
                    Stylometric, lexical and semantic evidence
                  </h3>
                </div>
              </div>

              <ParameterGrid parameters={parameters} />
            </section>
          )}
        </div>

        <div
          className="text-console-transition"
          key={displayedSentence?.id || "empty"}
        >
          <SentencePanel
            sentence={displayedSentence}
            locked={locked}
            onUnlock={() =>
              setSelectedSentence(null)
            }
            onVerifyLater={handleTextVerifyLater}
          />
        </div>
      </div>

      <section className="text-risk-legend">
        {[
          ["LOW", "0–39%", "Mostly natural linguistic variation."],
          ["MEDIUM", "40–69%", "Mixed evidence; manual review advised."],
          ["HIGH", "70–100%", "Strong AI-writing indicators detected."],
        ].map(([risk, range, explanation]) => (
          <article
            className={riskClass(risk)}
            key={risk}
          >
            <span />
            <div>
              <h4>{risk} RISK</h4>
              <strong>{range}</strong>
              <p>{explanation}</p>
            </div>
          </article>
        ))}
      </section>

      <p className="text-investigator-disclaimer">
        Sentence scores are explainable model indicators, not standalone proof
        of authorship. The final assessment should combine document-level and
        parameter-level evidence.
      </p>
    </section>
  );
}

export default InteractiveTextInvestigator;