import React, { useEffect, useMemo, useRef, useState } from "react";
import "./InteractiveImageInvestigator.css";

const VIEW_LABELS = {
  original: "Original",
  overlay: "Forensic Overlay",
  heatmap: "Heatmap",
  naturalness_map: "Naturalness",
  edge_map: "Edge Map",
  frequency_map: "Frequency Map",
};

const METRIC_LABELS = {
  texture: "Texture consistency",
  frequency: "Frequency anomaly",
  edge: "Edge continuity",
  noise: "Noise consistency",
  color: "Colour consistency",
  lighting: "Lighting consistency",
  compression: "Compression consistency",
  boundary: "Boundary blending",
  symmetry: "Facial symmetry",
  gradient: "Gradient behaviour",
  entropy: "Signal entropy",
  manipulation: "Manipulation traces",
  deep_learning: "Deep-learning evidence",
};

const DEFAULT_LEGEND = [
  ["natural", "Strong natural indicators", "0–20%", "Expected camera texture, noise and frequency behaviour."],
  ["low", "Low suspicion", "20–40%", "Minor deviations that may come from compression or normal processing."],
  ["mixed", "Mixed evidence", "40–60%", "Natural and synthetic indicators overlap; manual review is recommended."],
  ["high", "High suspicion", "60–80%", "Multiple forensic indicators suggest editing or synthetic generation."],
  ["critical", "Strong AI evidence", "80–100%", "Strong statistical patterns associated with generated imagery."],
].map(([key, label, range, description]) => ({ key, label, range, description }));

function clamp(value, min = 0, max = 100) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "0.0";
}

function titleCase(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeRisk(value, score = 0) {
  const risk = String(value || "").toUpperCase();
  if (risk.includes("CRITICAL") || risk.includes("VERY HIGH")) return "CRITICAL";
  if (risk.includes("HIGH")) return "HIGH";
  if (risk.includes("MEDIUM") || risk.includes("MODERATE")) return "MEDIUM";
  if (risk.includes("LOW")) return "LOW";
  if (score >= 80) return "CRITICAL";
  if (score >= 60) return "HIGH";
  if (score >= 40) return "MEDIUM";
  return "LOW";
}

function riskClass(risk, score) {
  return `risk-${normalizeRisk(risk, score).toLowerCase()}`;
}

function normalizeScore(region) {
  const values = [
    region?.forgery_score,
    region?.risk_score,
    region?.score,
    region?.confidence,
    region?.ai_probability,
    region?.suspicion_score,
  ];

  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) {
      return clamp(number <= 1 ? number * 100 : number);
    }
  }

  return 0;
}

function imageDimensions(regionAnalysis) {
  const source =
    regionAnalysis?.image_dimensions ||
    regionAnalysis?.original_dimensions ||
    regionAnalysis?.dimensions ||
    {};

  return {
    width:
      Number(
        source.width ||
          source.image_width ||
          regionAnalysis?.image_width
      ) || 0,
    height:
      Number(
        source.height ||
          source.image_height ||
          regionAnalysis?.image_height
      ) || 0,
  };
}

function parseBox(region, dimensions) {
  const source =
    region?.bbox ||
    region?.box ||
    region?.coordinates ||
    region?.rectangle ||
    {};

  let x = source.x ?? source.left ?? source.x1 ?? region?.x ?? region?.left ?? region?.x1;
  let y = source.y ?? source.top ?? source.y1 ?? region?.y ?? region?.top ?? region?.y1;
  let width = source.width ?? source.w ?? region?.width ?? region?.w;
  let height = source.height ?? source.h ?? region?.height ?? region?.h;

  const x2 = source.x2 ?? source.right ?? region?.x2 ?? region?.right;
  const y2 = source.y2 ?? source.bottom ?? region?.y2 ?? region?.bottom;

  if (width == null && x != null && x2 != null) width = Number(x2) - Number(x);
  if (height == null && y != null && y2 != null) height = Number(y2) - Number(y);

  if (Array.isArray(source) && source.length >= 4) {
    [x, y, width, height] = source;
  }

  x = Number(x);
  y = Number(y);
  width = Number(width);
  height = Number(height);

  if (![x, y, width, height].every(Number.isFinite)) return null;
  if (width <= 0 || height <= 0) return null;

  const normalized =
    x >= 0 &&
    y >= 0 &&
    x <= 1.001 &&
    y <= 1.001 &&
    width <= 1.001 &&
    height <= 1.001;

  if (normalized) {
    return {
      x: x * 100,
      y: y * 100,
      width: width * 100,
      height: height * 100,
    };
  }

  const percentages =
    x >= 0 &&
    y >= 0 &&
    x <= 100.001 &&
    y <= 100.001 &&
    width <= 100.001 &&
    height <= 100.001;

  if (percentages) {
    return { x, y, width, height };
  }

  if (dimensions.width > 0 && dimensions.height > 0) {
    return {
      x: (x / dimensions.width) * 100,
      y: (y / dimensions.height) * 100,
      width: (width / dimensions.width) * 100,
      height: (height / dimensions.height) * 100,
    };
  }

  return null;
}

function normalizeRegions(regionAnalysis) {
  const source =
    regionAnalysis?.ranked_regions ||
    regionAnalysis?.regions ||
    regionAnalysis?.facial_regions ||
    regionAnalysis?.patches ||
    [];

  const dimensions = imageDimensions(regionAnalysis);

  return safeArray(source)
    .map((region, index) => {
      const score = normalizeScore(region);
      const reasons = safeArray(
        region?.reasons ||
          region?.findings ||
          region?.indicators ||
          region?.evidence
      ).filter(Boolean);

      return {
        ...region,
        id:
          region?.id ||
          region?.region_id ||
          region?.name ||
          region?.label ||
          `region-${index}`,
        name:
          region?.name ||
          region?.label ||
          region?.region_name ||
          `Region ${index + 1}`,
        score,
        risk: normalizeRisk(
          region?.risk_level ||
            region?.risk ||
            region?.classification,
          score
        ),
        box: parseBox(region, dimensions),
        reasons,
        metrics:
          region?.metrics ||
          region?.feature_scores ||
          region?.indicators_by_score ||
          {},
        interpretation:
          region?.interpretation ||
          region?.explanation ||
          region?.reason ||
          reasons[0] ||
          "This region contains measurable forensic deviations that should be reviewed with nearby regions and the global model result.",
        recommendation:
          region?.recommendation ||
          "Inspect this region with adjacent facial boundaries and the frequency-domain layer.",
      };
    })
    .filter((region) => region.box)
    .sort((a, b) => b.score - a.score);
}

function pointInside(point, box) {
  return (
    box &&
    point.x >= box.x &&
    point.x <= box.x + box.width &&
    point.y >= box.y &&
    point.y <= box.y + box.height
  );
}

function findRegionAtPoint(regions, point) {
  return regions
    .filter((region) => pointInside(point, region.box))
    .sort(
      (a, b) =>
        a.box.width * a.box.height -
        b.box.width * b.box.height
    )[0];
}

function buildViews(originalUrl, visualEvidence) {
  return [
    ["original", originalUrl],
    ["overlay", visualEvidence?.overlay],
    ["heatmap", visualEvidence?.heatmap],
    ["naturalness_map", visualEvidence?.naturalness_map],
    ["edge_map", visualEvidence?.edge_map],
    ["frequency_map", visualEvidence?.frequency_map],
  ]
    .filter(([, url]) => url)
    .map(([key, url]) => ({ key, url }));
}

function MetricBars({ metrics }) {
  const entries = Object.entries(metrics || {})
    .map(([key, raw]) => {
      const value =
        typeof raw === "object"
          ? raw?.score ?? raw?.value ?? 0
          : raw;

      const number = Number(value);

      return {
        key,
        label: METRIC_LABELS[key] || titleCase(key),
        score: clamp(number <= 1 ? number * 100 : number),
      };
    })
    .filter((item) => Number.isFinite(item.score))
    .sort((a, b) => b.score - a.score);

  if (!entries.length) {
    return (
      <p className="image-empty-copy">
        No metric-level evidence was returned for this region.
      </p>
    );
  }

  return (
    <div className="image-metric-list">
      {entries.map((item) => (
        <div className="image-metric-row" key={item.key}>
          <div>
            <span>{item.label}</span>
            <strong>{formatNumber(item.score)}%</strong>
          </div>
          <section>
            <i style={{ width: `${item.score}%` }} />
          </section>
        </div>
      ))}
    </div>
  );
}

function RegionConsole({ region, locked, onUnlock, onVerifyLater }) {
  if (!region) {
    return (
      <aside className="image-live-console image-live-console-empty">
        <div className="image-console-radar">
          <span />
        </div>
        <p className="image-eyebrow">LIVE REGION ANALYSIS</p>
        <h3>Move over the evidence</h3>
        <p>
          Hover for a temporary preview. Click a region to lock its
          forensic findings.
        </p>
        {/* CONTINUOUS LEARNING & VERIFY LATER FEEDBACK BLOCK */}
        <section className="image-console-section" style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
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

  return (
    <aside className="image-live-console">
      <div className="image-console-heading">
        <div>
          <p className="image-eyebrow">
            {locked
              ? "LOCKED REGION ANALYSIS"
              : "LIVE REGION ANALYSIS"}
          </p>
          <h3>{region.name}</h3>
        </div>
        <span className={riskClass(region.risk, region.score)}>
          {region.risk}
        </span>
      </div>

      <div className="image-region-score">
        <span>Forgery suspicion</span>
        <strong>{formatNumber(region.score)}%</strong>
        <div>
          <i style={{ width: `${region.score}%` }} />
        </div>
      </div>

      <div className="image-console-facts">
        <article>
          <span>Region</span>
          <strong>{region.name}</strong>
        </article>
        <article>
          <span>Risk</span>
          <strong>{region.risk}</strong>
        </article>
        <article>
          <span>Mode</span>
          <strong>{locked ? "Locked" : "Preview"}</strong>
        </article>
      </div>

      <section className="image-console-section">
        <h4>Forensic indicators</h4>
        <MetricBars metrics={region.metrics} />
      </section>

      <section className="image-console-section">
        <h4>Detected findings</h4>
        {region.reasons.length ? (
          <ul className="image-findings-list">
            {region.reasons.slice(0, 6).map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        ) : (
          <p className="image-empty-copy">
            No text findings were returned.
          </p>
        )}
      </section>

      <section className="image-console-section">
        <h4>AI interpretation</h4>
        <p>{region.interpretation}</p>
      </section>

      <section className="image-console-section image-console-recommendation">
        <h4>Recommended review</h4>
        <p>{region.recommendation}</p>
      </section>

      {locked && (
        <button
          type="button"
          className="image-unlock-button"
          onClick={onUnlock}
        >
          Unlock region
        </button>
      )}
    </aside>
  );
}

function HeatmapLegend({ legend }) {
  const supplied = safeArray(legend);

  const entries = supplied.length
    ? supplied.map((item, index) =>
        typeof item === "string"
          ? {
              key: `legend-${index}`,
              label: item,
              range: "",
              description: "",
            }
          : {
              key: item?.key || `legend-${index}`,
              label:
                item?.label ||
                item?.name ||
                item?.title ||
                "Indicator",
              range: item?.range || item?.score_range || "",
              description:
                item?.description || item?.meaning || "",
            }
      )
    : DEFAULT_LEGEND;

  return (
    <section className="image-legend-panel">
      <div className="image-section-heading">
        <div>
          <p className="image-eyebrow">HEATMAP INTERPRETATION</p>
          <h3>What each colour means</h3>
        </div>
        <span>Evidence guide</span>
      </div>

      <div className="image-legend-grid">
        {entries.map((entry) => (
          <article
            className={`image-legend-card legend-${entry.key}`}
            key={entry.key}
          >
            <span className="image-legend-swatch" />
            <div>
              <h4>{entry.label}</h4>
              {entry.range && <strong>{entry.range}</strong>}
              {entry.description && <p>{entry.description}</p>}
            </div>
          </article>
        ))}
      </div>

      <p className="image-legend-notice">
        Colours represent relative anomaly intensity. They are not
        pixel-level proof of manipulation.
      </p>
    </section>
  );
}

function RegionExplorer({ regions, selectedId, onSelect }) {
  if (!regions.length) return null;

  return (
    <section className="image-region-explorer">
      <div className="image-section-heading">
        <div>
          <p className="image-eyebrow">REGION EXPLORER</p>
          <h3>Ranked facial and patch analysis</h3>
        </div>
        <span>{regions.length} regions</span>
      </div>

      <div className="image-region-grid">
        {regions.map((region, index) => (
          <button
            type="button"
            className={[
              "image-region-card",
              riskClass(region.risk, region.score),
              selectedId === region.id ? "active" : "",
            ].join(" ")}
            key={region.id}
            onClick={() => onSelect(region)}
          >
            <span className="image-region-rank">
              {String(index + 1).padStart(2, "0")}
            </span>

            <div className="image-region-card-copy">
              <div>
                <strong>{region.name}</strong>
                <small>{region.risk}</small>
              </div>
              <section>
                <i style={{ width: `${region.score}%` }} />
              </section>
            </div>

            <b>{formatNumber(region.score)}%</b>
          </button>
        ))}
      </div>
    </section>
  );
}

function EvidenceSummary({ regions, regionAnalysis }) {
  const overall = clamp(
    regionAnalysis?.overall_region_score ??
      regionAnalysis?.overall_risk_score ??
      regionAnalysis?.risk_score ??
      (regions.length
        ? regions.reduce((sum, region) => sum + region.score, 0) /
          regions.length
        : 0)
  );

  const findings = safeArray(
    regionAnalysis?.reasons ||
      regionAnalysis?.global_findings ||
      regionAnalysis?.findings
  );

  return (
    <section className="image-evidence-summary">
      <div className="image-section-heading">
        <div>
          <p className="image-eyebrow">FORENSIC EVIDENCE SUMMARY</p>
          <h3>Global image interpretation</h3>
        </div>
      </div>

      <div className="image-summary-grid">
        <article className="image-overall-risk-card">
          <span>Regional suspicion index</span>
          <strong>{formatNumber(overall)}%</strong>
          <div>
            <i style={{ width: `${overall}%` }} />
          </div>
          <p>
            This aggregates regional findings; use the main model
            output for the final verdict.
          </p>
        </article>

        <article>
          <h4>Primary evidence</h4>
          {findings.length ? (
            <ul className="image-summary-list">
              {findings.slice(0, 7).map((finding) => (
                <li key={finding}>{finding}</li>
              ))}
            </ul>
          ) : (
            <p>No global findings were returned.</p>
          )}
        </article>

        <article>
          <h4>Most suspicious regions</h4>
          <ol className="image-summary-ranking">
            {regions.slice(0, 5).map((region) => (
              <li key={region.id}>
                <span>{region.name}</span>
                <strong>{formatNumber(region.score)}%</strong>
              </li>
            ))}
          </ol>
        </article>
      </div>
    </section>
  );
}

function InteractiveImageInvestigator({
  originalUrl,
  visualEvidence = {},
  regionAnalysis = {},
}) {
  const imageFrameRef = useRef(null);
  const [activeView, setActiveView] = useState("heatmap");
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [showRegions, setShowRegions] = useState(true);
  const [pointer, setPointer] = useState({
    visible: false,
    x: 0,
    y: 0,
  });

  const views = useMemo(
    () => buildViews(originalUrl, visualEvidence),
    [originalUrl, visualEvidence]
  );

  const regions = useMemo(
    () => normalizeRegions(regionAnalysis),
    [regionAnalysis]
  );

  useEffect(() => {
    const keys = views.map((view) => view.key);

    if (!keys.includes(activeView)) {
      setActiveView(
        keys.includes("heatmap")
          ? "heatmap"
          : keys[0]
      );
    }
  }, [activeView, views]);

  useEffect(() => {
    setHoveredRegion(null);
    setSelectedRegion(null);
  }, [regionAnalysis]);

  const active =
    views.find((view) => view.key === activeView) ||
    views[0];

  const displayedRegion =
    hoveredRegion ||
    selectedRegion;

  const locked =
    Boolean(
      selectedRegion &&
      !hoveredRegion
    );

  function eventPoint(event) {
    const image =
      imageFrameRef.current?.querySelector(
        ".forensic-evidence-image"
      );

    if (!image) return null;

    const bounds =
      image.getBoundingClientRect();

    const x =
      event.clientX -
      bounds.left;

    const y =
      event.clientY -
      bounds.top;

    if (
      x < 0 ||
      y < 0 ||
      x > bounds.width ||
      y > bounds.height
    ) {
      return null;
    }

    return {
      x: (x / bounds.width) * 100,
      y: (y / bounds.height) * 100,
    };
  }

  function handleMove(event) {
    const point = eventPoint(event);

    if (!point) {
      setPointer((current) => ({
        ...current,
        visible: false,
      }));
      setHoveredRegion(null);
      return;
    }

    setPointer({
      visible: true,
      x: point.x,
      y: point.y,
    });

    setHoveredRegion(
      showRegions
        ? findRegionAtPoint(regions, point) || null
        : null
    );
  }

  function handleClick(event) {
    const point = eventPoint(event);

    if (!point) {
      setSelectedRegion(null);
      return;
    }

    setSelectedRegion(
      findRegionAtPoint(regions, point) || null
    );
  }

  async function handleVerifyLater(e) {
    if (e && e.preventDefault) e.preventDefault();
    try {
      const response = await fetch("http://localhost:8000/verify-later", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modality: "image",
          identifier: originalUrl || "Image_Evidence",
          prediction: regionAnalysis?.prediction || "Manipulated",
          confidence: regionAnalysis?.overall_region_score || 0.85,
          filepath: originalUrl,
        }),
      });

      if (response.ok) {
        alert("Image sample moved to Pending Verification!");
      } else {
        alert("Failed to move item to pending verification.");
      }
    } catch (err) {
      console.error("Error staging image for verification:", err);
    }
  }

  if (!active?.url) return null;

  return (
    <section className="image-investigator">
      <header className="image-investigator-header">
        <div>
          <p className="image-eyebrow">
            FORGE IMAGE XAI 3.0
          </p>
          <h2>
            Interactive Image Forensic Workstation
          </h2>
          <p>
            Inspect visual layers, hover for temporary evidence and
            click to lock a finding.
          </p>
        </div>

        <div className="image-investigator-status">
          <span>{regions.length} analysed regions</span>
          <span>
            {regionAnalysis?.face_detected
              ? "Face detected"
              : "Patch analysis"}
          </span>
          <span>
            {regionAnalysis?.analysis_version ||
              "Region XAI"}
          </span>
        </div>
      </header>

      <nav className="image-view-tabs">
        <div>
          {views.map((view) => (
            <button
              type="button"
              className={
                activeView === view.key
                  ? "active"
                  : ""
              }
              key={view.key}
              onClick={() =>
                setActiveView(view.key)
              }
            >
              {VIEW_LABELS[view.key] ||
                titleCase(view.key)}
            </button>
          ))}
        </div>

        <button
          type="button"
          className={`image-region-toggle ${
            showRegions ? "active" : ""
          }`}
          onClick={() =>
            setShowRegions((value) => !value)
          }
        >
          {showRegions
            ? "Hide region boxes"
            : "Show region boxes"}
        </button>
      </nav>

      <div className="image-workstation-grid">
        <div className="image-evidence-column">
          <div className="forensic-image-stage">
            <div
              ref={imageFrameRef}
              className="forensic-image-frame"
              onPointerMove={handleMove}
              onPointerLeave={() => {
                setHoveredRegion(null);
                setPointer((current) => ({
                  ...current,
                  visible: false,
                }));
              }}
              onClick={handleClick}
            >
              <img
                src={active.url}
                alt={`${
                  VIEW_LABELS[active.key] ||
                  active.key
                } forensic layer`}
                className={`forensic-evidence-image view-${active.key}`}
                draggable="false"
              />

              {showRegions && (
                <div className="forensic-region-layer">
                  {regions.map((region) => {
                    const hovered =
                      hoveredRegion?.id === region.id;

                    const selected =
                      selectedRegion?.id === region.id;

                    return (
                      <div
                        className={[
                          "forensic-region-box",
                          riskClass(
                            region.risk,
                            region.score
                          ),
                          hovered
                            ? "hovered"
                            : "",
                          selected
                            ? "selected"
                            : "",
                        ].join(" ")}
                        key={region.id}
                        style={{
                          left: `${region.box.x}%`,
                          top: `${region.box.y}%`,
                          width: `${region.box.width}%`,
                          height: `${region.box.height}%`,
                        }}
                      />
                    );
                  })}
                </div>
              )}

              <div className="image-scan-line" />

              {pointer.visible && (
                <div
                  className="image-live-cursor"
                  style={{
                    left: `${pointer.x}%`,
                    top: `${pointer.y}%`,
                  }}
                >
                  <span />
                  <i />
                  <b />
                </div>
              )}

              {hoveredRegion?.box && (
                <svg
                  className="image-region-connector"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  aria-hidden="true"
                >
                  <line
                    x1={pointer.x}
                    y1={pointer.y}
                    x2="100"
                    y2="50"
                  />
                </svg>
              )}

              <div className="image-stage-badge">
                {VIEW_LABELS[active.key] ||
                  titleCase(active.key)}
              </div>

              <div className="image-stage-help">
                Hover to preview • Click to lock
              </div>
            </div>
          </div>

          {visualEvidence?.interpretation_notice && (
            <p className="image-interpretation-notice">
              {
                visualEvidence.interpretation_notice
              }
            </p>
          )}
        </div>

        <div
          className="image-console-transition"
          key={displayedRegion?.id || "empty"}
        >
          <RegionConsole
            region={displayedRegion}
            locked={locked}
            onUnlock={() =>
              setSelectedRegion(null)
            }
            onVerifyLater={handleVerifyLater}
          />
        </div>
      </div>

      <HeatmapLegend
        legend={visualEvidence?.legend}
      />

      <RegionExplorer
        regions={regions}
        selectedId={selectedRegion?.id}
        onSelect={(region) => {
          setSelectedRegion(region);
          setHoveredRegion(null);
        }}
      />

      <EvidenceSummary
        regions={regions}
        regionAnalysis={regionAnalysis}
      />

      <p className="image-investigator-disclaimer">
        Region boxes are aligned to the actual rendered image rather
        than the outer black container. Regional scores are
        explainable indicators, not standalone proof.
      </p>
    </section>
  );
}

export default InteractiveImageInvestigator;