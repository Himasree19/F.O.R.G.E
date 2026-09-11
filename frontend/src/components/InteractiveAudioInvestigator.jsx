import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./InteractiveAudioInvestigator.css";


const SEGMENT_METRIC_LABELS = {
  pitch: "Pitch Irregularity",
  energy: "Energy Variation",
  spectral_flatness: "Spectral Flatness",
  zero_crossing: "Zero-Crossing Pattern",
  pause_pattern: "Pause Pattern",
  phase: "Phase Discontinuity",
  noise: "Noise Inconsistency",
  frequency: "Frequency Anomaly",
  bandwidth: "Spectral Bandwidth",
  amplitude: "Amplitude Discontinuity",
  entropy: "Signal Entropy",
  voice_stability: "Voice Stability",
};


const VOICE_DNA_LABELS = {
  naturalness: "Voice Naturalness",
  synthetic_signature: "Synthetic Signature",
  pitch_stability: "Pitch Stability",
  prosody_risk: "Prosody Risk",
  temporal_consistency: "Temporal Consistency",
  breathing_naturalness: "Breathing Naturalness",
  jitter: "Jitter",
  shimmer: "Shimmer",
  model_synthetic_probability: "Model Synthetic Probability",
};


function clamp(value, minimum, maximum) {
  return Math.min(
    maximum,
    Math.max(
      minimum,
      Number(value) || 0
    )
  );
}


function riskClass(risk = "LOW") {
  return `risk-${String(risk).toLowerCase()}`;
}


function formatTime(seconds) {
  const safeSeconds = Math.max(
    0,
    Number(seconds) || 0
  );

  const minutes = Math.floor(
    safeSeconds / 60
  );

  const remaining =
    safeSeconds % 60;

  return `${String(minutes).padStart(2, "0")}:${remaining
    .toFixed(2)
    .padStart(5, "0")}`;
}


function formatNumber(
  value,
  digits = 2
) {
  const number =
    Number(value);

  if (!Number.isFinite(number)) {
    return "0.00";
  }

  return number.toFixed(
    digits
  );
}


function prepareCurveData(data, duration) {
  if (!Array.isArray(data)) {
    return [];
  }

  const cleaned = data
    .map((item, index) => {
      const time = Number(item?.time);
      const value = Number(item?.value);

      if (!Number.isFinite(time) || !Number.isFinite(value)) {
        return null;
      }

      return {
        id: index,
        time,
        value,
      };
    })
    .filter(Boolean)
    .filter(
      (item) =>
        item.time >= 0 &&
        (!Number.isFinite(Number(duration)) ||
          Number(duration) <= 0 ||
          item.time <= Number(duration) + 0.5)
    )
    .sort((first, second) => first.time - second.time);

  return cleaned;
}


function CurveChart({
  title,
  subtitle,
  data,
  currentTime,
  duration,
  unit,
  onSeek,
}) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);

  const [hoverPoint, setHoverPoint] = useState(null);
  const [canvasSize, setCanvasSize] = useState({
    width: 900,
    height: 270,
  });

  const chartData = useMemo(
    () => prepareCurveData(data, duration),
    [data, duration]
  );

  const statistics = useMemo(() => {
    if (!chartData.length) {
      return {
        minimum: 0,
        maximum: 0,
        average: 0,
        timeMaximum: Math.max(Number(duration) || 0, 1),
      };
    }

    const values = chartData.map((item) => item.value);

    return {
      minimum: Math.min(...values),
      maximum: Math.max(...values),
      average:
        values.reduce((total, value) => total + value, 0) /
        values.length,
      timeMaximum: Math.max(
        Number(duration) || 0,
        chartData[chartData.length - 1]?.time || 0,
        1
      ),
    };
  }, [chartData, duration]);

  useEffect(() => {
    const wrapper = wrapperRef.current;

    if (!wrapper) {
      return undefined;
    }

    const updateSize = () => {
      const bounds = wrapper.getBoundingClientRect();

      setCanvasSize({
        width: Math.max(320, Math.floor(bounds.width || 900)),
        height: 270,
      });
    };

    updateSize();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize);

      return () => {
        window.removeEventListener("resize", updateSize);
      };
    }

    const observer = new ResizeObserver(updateSize);
    observer.observe(wrapper);

    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    const ratio = window.devicePixelRatio || 1;
    const width = canvasSize.width;
    const height = canvasSize.height;

    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const padding = {
      top: 20,
      right: 18,
      bottom: 42,
      left: 70,
    };

    const plotWidth = Math.max(
      1,
      width - padding.left - padding.right
    );

    const plotHeight = Math.max(
      1,
      height - padding.top - padding.bottom
    );

    context.fillStyle = "#020711";
    context.fillRect(0, 0, width, height);

    const topGradient = context.createLinearGradient(
      0,
      padding.top,
      0,
      padding.top + plotHeight
    );

    topGradient.addColorStop(0, "rgba(0,229,255,0.065)");
    topGradient.addColorStop(1, "rgba(0,229,255,0)");

    context.fillStyle = topGradient;
    context.fillRect(
      padding.left,
      padding.top,
      plotWidth,
      plotHeight
    );

    if (chartData.length < 2) {
      context.fillStyle = "#79849f";
      context.font = "12px Arial";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(
        `No valid ${title.toLowerCase()} points received`,
        width / 2,
        height / 2
      );
      return;
    }

    let valueMinimum = statistics.minimum;
    let valueMaximum = statistics.maximum;
    let valueRange = valueMaximum - valueMinimum;

    if (!Number.isFinite(valueRange) || Math.abs(valueRange) < 1e-12) {
      const fallback = Math.max(
        Math.abs(valueMaximum) * 0.12,
        unit === "Hz" ? 5 : 0.001
      );

      valueMinimum -= fallback;
      valueMaximum += fallback;
    } else {
      const extra = valueRange * 0.1;
      valueMinimum -= extra;
      valueMaximum += extra;
    }

    valueRange = Math.max(valueMaximum - valueMinimum, 1e-12);
    const timeMaximum = statistics.timeMaximum;

    const xForTime = (time) =>
      padding.left +
      (clamp(time, 0, timeMaximum) / timeMaximum) * plotWidth;

    const yForValue = (value) =>
      padding.top +
      plotHeight -
      ((value - valueMinimum) / valueRange) * plotHeight;

    context.strokeStyle = "rgba(255,255,255,0.08)";
    context.lineWidth = 1;
    context.font = "10px Arial";
    context.fillStyle = "#75809b";

    const horizontalLines = 5;

    for (let index = 0; index <= horizontalLines; index += 1) {
      const fraction = index / horizontalLines;
      const y = padding.top + fraction * plotHeight;
      const labelValue = valueMaximum - fraction * valueRange;

      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(padding.left + plotWidth, y);
      context.stroke();

      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(
        labelValue.toFixed(unit === "Hz" ? 0 : 5),
        padding.left - 9,
        y
      );
    }

    const verticalLines = 4;

    for (let index = 0; index <= verticalLines; index += 1) {
      const fraction = index / verticalLines;
      const x = padding.left + fraction * plotWidth;

      context.beginPath();
      context.moveTo(x, padding.top);
      context.lineTo(x, padding.top + plotHeight);
      context.stroke();

      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(
        formatTime(fraction * timeMaximum),
        x,
        padding.top + plotHeight + 10
      );
    }

    const areaGradient = context.createLinearGradient(
      0,
      padding.top,
      0,
      padding.top + plotHeight
    );

    areaGradient.addColorStop(0, "rgba(0,229,255,0.22)");
    areaGradient.addColorStop(1, "rgba(122,92,255,0.01)");

    context.beginPath();

    chartData.forEach((point, index) => {
      const x = xForTime(point.time);
      const y = yForValue(point.value);

      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });

    context.lineTo(
      xForTime(chartData[chartData.length - 1].time),
      padding.top + plotHeight
    );
    context.lineTo(
      xForTime(chartData[0].time),
      padding.top + plotHeight
    );
    context.closePath();
    context.fillStyle = areaGradient;
    context.fill();

    const lineGradient = context.createLinearGradient(
      padding.left,
      0,
      padding.left + plotWidth,
      0
    );

    lineGradient.addColorStop(0, "#00e5ff");
    lineGradient.addColorStop(0.5, "#7a5cff");
    lineGradient.addColorStop(1, "#ff3d71");

    context.beginPath();

    chartData.forEach((point, index) => {
      const x = xForTime(point.time);
      const y = yForValue(point.value);

      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });

    context.strokeStyle = lineGradient;
    context.lineWidth = 2.7;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();

    const playheadX = xForTime(Number(currentTime) || 0);

    context.beginPath();
    context.moveTo(playheadX, padding.top);
    context.lineTo(playheadX, padding.top + plotHeight);
    context.strokeStyle = "rgba(255,255,255,0.95)";
    context.lineWidth = 2;
    context.stroke();

    if (hoverPoint) {
      const hoverX = xForTime(hoverPoint.time);
      const hoverY = yForValue(hoverPoint.value);

      context.beginPath();
      context.moveTo(hoverX, padding.top);
      context.lineTo(hoverX, padding.top + plotHeight);
      context.strokeStyle = "rgba(0,229,255,0.55)";
      context.lineWidth = 1;
      context.stroke();

      context.beginPath();
      context.arc(hoverX, hoverY, 5, 0, Math.PI * 2);
      context.fillStyle = "#00e5ff";
      context.fill();
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2;
      context.stroke();
    }
  }, [
    canvasSize,
    chartData,
    currentTime,
    hoverPoint,
    statistics,
    title,
    unit,
  ]);

  function findNearestPoint(event) {
    const canvas = canvasRef.current;

    if (!canvas || !chartData.length) {
      return null;
    }

    const bounds = canvas.getBoundingClientRect();
    const paddingLeft = 70;
    const paddingRight = 18;
    const plotWidth = Math.max(
      1,
      bounds.width - paddingLeft - paddingRight
    );

    const localX = clamp(
      event.clientX - bounds.left - paddingLeft,
      0,
      plotWidth
    );

    const targetTime =
      (localX / plotWidth) * statistics.timeMaximum;

    let nearest = chartData[0];
    let distance = Math.abs(nearest.time - targetTime);

    for (let index = 1; index < chartData.length; index += 1) {
      const candidate = chartData[index];
      const candidateDistance = Math.abs(candidate.time - targetTime);

      if (candidateDistance < distance) {
        nearest = candidate;
        distance = candidateDistance;
      }
    }

    return nearest;
  }

  function handleMouseMove(event) {
    setHoverPoint(findNearestPoint(event));
  }

  function handleMouseLeave() {
    setHoverPoint(null);
  }

  function handleClick(event) {
    const nearest = findNearestPoint(event);

    if (nearest && typeof onSeek === "function") {
      onSeek(nearest.time);
    }
  }

  return (
    <section className="audio-curve-card">
      <div className="audio-curve-header">
        <div>
          <p>ACOUSTIC CURVE</p>
          <h3>{title}</h3>
          <span>{subtitle}</span>
        </div>

        <div>
          {chartData.length} points
        </div>
      </div>

      <div
        ref={wrapperRef}
        className="audio-canvas-wrapper"
        style={{
          position: "relative",
          width: "100%",
          minHeight: "270px",
          overflow: "hidden",
          borderRadius: "14px",
          background: "#020711",
          border: "1px solid rgba(255,255,255,0.055)",
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          style={{
            display: "block",
            width: "100%",
            height: "270px",
            cursor: "crosshair",
          }}
        />

        {hoverPoint && (
          <div
            className="audio-canvas-tooltip"
            style={{
              position: "absolute",
              top: "14px",
              right: "14px",
              zIndex: 5,
              minWidth: "160px",
              padding: "11px 13px",
              borderRadius: "12px",
              color: "#ffffff",
              background: "rgba(3,8,22,0.96)",
              border: "1px solid rgba(0,229,255,0.3)",
              boxShadow: "0 16px 45px rgba(0,0,0,0.5)",
              pointerEvents: "none",
            }}
          >
            <span style={{ display: "block", color: "#8c97b5", fontSize: "10px" }}>
              {formatTime(hoverPoint.time)}
            </span>

            <strong style={{ display: "block", marginTop: "5px", color: "#00e5ff", fontSize: "17px" }}>
              {formatNumber(
                hoverPoint.value,
                unit === "Hz" ? 2 : 6
              )}{" "}
              {unit}
            </strong>

            <small style={{ display: "block", marginTop: "6px", color: "#747f9b", fontSize: "9px" }}>
              Click to seek audio
            </small>
          </div>
        )}
      </div>

      <div className="audio-curve-footer">
        <span>
          Minimum: {formatNumber(statistics.minimum, unit === "Hz" ? 1 : 6)} {unit}
        </span>

        <span>
          Average: {formatNumber(statistics.average, unit === "Hz" ? 1 : 6)} {unit}
        </span>

        <span>
          Maximum: {formatNumber(statistics.maximum, unit === "Hz" ? 1 : 6)} {unit}
        </span>

        <span>
          Current: {formatTime(currentTime)}
        </span>
      </div>
    </section>
  );
}


function VoiceDNAPanel({
  voiceDNA,
}) {
  const entries =
    Object.entries(
      voiceDNA || {}
    );

  if (!entries.length) {
    return null;
  }

  return (
    <section className="voice-dna-panel">

      <div className="voice-dna-header">

        <div>
          <p>
            VOICE DNA PROFILE
          </p>

          <h3>
            Acoustic Identity and Synthetic Signature
          </h3>
        </div>

        <span>
          {entries.length} indicators
        </span>

      </div>


      <div className="voice-dna-grid">

        {entries.map(
          ([
            key,
            value,
          ]) => {
            const score =
              clamp(
                value?.score,
                0,
                100
              );

            return (
              <article
                className={[
                  "voice-dna-card",
                  riskClass(
                    value?.risk
                  ),
                ].join(" ")}
                key={key}
              >

                <div className="voice-dna-card-head">

                  <div>
                    <span>
                      {
                        VOICE_DNA_LABELS[
                          key
                        ]
                        || key
                      }
                    </span>

                    <strong>
                      {formatNumber(
                        score
                      )}%
                    </strong>
                  </div>

                  <b>
                    {
                      value?.risk
                      || "LOW"
                    }
                  </b>

                </div>


                <div className="voice-dna-track">
                  <i
                    style={{
                      width:
                        `${score}%`,
                    }}
                  />
                </div>


                {value?.observed !== undefined && (
                  <p className="voice-dna-observed">
                    Observed value:{" "}
                    <strong>
                      {
                        formatNumber(
                          value.observed,
                          6
                        )
                      }
                    </strong>
                  </p>
                )}


                <p>
                  {
                    value?.reason
                    || "No explanation available."
                  }
                </p>

              </article>
            );
          }
        )}

      </div>

    </section>
  );
}


function AudioSummaryPanel({
  summary,
}) {
  if (
    !summary
    || !Object.keys(summary).length
  ) {
    return null;
  }

  const summaryItems = [
    {
      label:
        "Mean Pitch",
      value:
        `${formatNumber(
          summary.pitch_mean_hz
        )} Hz`,
    },
    {
      label:
        "Pitch Variation",
      value:
        `${formatNumber(
          summary.pitch_std_hz
        )} Hz`,
    },
    {
      label:
        "Mean Energy",
      value:
        formatNumber(
          summary.energy_mean,
          6
        ),
    },
    {
      label:
        "Energy Variation",
      value:
        formatNumber(
          summary.energy_variation,
          6
        ),
    },
    {
      label:
        "Pause Count",
      value:
        summary.pause_count
        ?? 0,
    },
    {
      label:
        "Pause Ratio",
      value:
        `${formatNumber(
          (
            Number(
              summary.pause_ratio
              || 0
            )
            * 100
          )
        )}%`,
    },
    {
      label:
        "Breathing Events",
      value:
        summary
          .breathing_event_count
        ?? 0,
    },
    {
      label:
        "Estimated Breaths/Min",
      value:
        formatNumber(
          summary
            .estimated_breaths_per_minute
        ),
    },
  ];


  return (
    <section className="audio-summary-panel">

      <div className="audio-summary-header">

        <div>
          <p>
            ADVANCED ACOUSTIC SUMMARY
          </p>

          <h3>
            Voice and Signal Statistics
          </h3>
        </div>

      </div>


      <div className="audio-summary-grid">

        {summaryItems.map(
          (item) => (
            <article
              key={
                item.label
              }
            >

              <span>
                {item.label}
              </span>

              <strong>
                {item.value}
              </strong>

            </article>
          )
        )}

      </div>

    </section>
  );
}


function InteractiveAudioInvestigator({
  audioUrl,
  audioTimeline = {},
  waveformUrl = null,
  spectrogramUrl = null,
  heatmapUrl = null,
  voiceDNA = {},
  audioCurves = {},
  audioSummary = {},
  pauseIntervals = [],
  breathingEvents = [],
}) {
  const audioRef =
    useRef(null);

  const timelineRef =
    useRef(null);

  const [
    currentTime,
    setCurrentTime,
  ] = useState(0);

  const [
    duration,
    setDuration,
  ] = useState(
    Number(
      audioTimeline
        ?.audio_duration
      || audioSummary
        ?.duration_seconds
      || 0
    )
  );

  const [
    isPlaying,
    setIsPlaying,
  ] = useState(false);

  const [
    activeSegment,
    setActiveSegment,
  ] = useState(null);

  const [
    selectedSegment,
    setSelectedSegment,
  ] = useState(null);

  const [
    investigationLog,
    setInvestigationLog,
  ] = useState([]);

  const [
    tooltipPosition,
    setTooltipPosition,
  ] = useState({
    x: 20,
    y: 20,
  });


  const segments =
    useMemo(
      () =>
        audioTimeline
          ?.segments
        || [],
      [audioTimeline]
    );


  const rankedSegments =
    useMemo(
      () =>
        audioTimeline
          ?.ranked_segments
        || [...segments]
          .sort(
            (
              first,
              second
            ) =>
              Number(
                second
                  .risk_score
                || 0
              )
              -
              Number(
                first
                  .risk_score
                || 0
              )
          )
          .slice(
            0,
            20
          ),
      [
        audioTimeline,
        segments,
      ]
    );


  const suspiciousIntervals =
    useMemo(
      () =>
        audioTimeline
          ?.suspicious_intervals
        || [],
      [audioTimeline]
    );


  const normalizedPauseIntervals =
    Array.isArray(
      pauseIntervals
    )
      ? pauseIntervals
      : [];


  const normalizedBreathingEvents =
    Array.isArray(
      breathingEvents
    )
      ? breathingEvents
      : [];


  useEffect(() => {
    setDuration(
      Number(
        audioTimeline
          ?.audio_duration
        || audioSummary
          ?.duration_seconds
        || 0
      )
    );
  }, [
    audioTimeline,
    audioSummary,
  ]);


  useEffect(() => {
    const audio =
      audioRef.current;

    if (!audio) {
      return undefined;
    }


    function handleTimeUpdate() {
      setCurrentTime(
        audio.currentTime
        || 0
      );
    }


    function handleMetadata() {
      setDuration(
        audio.duration
        || Number(
          audioTimeline
            ?.audio_duration
          || 0
        )
      );
    }


    function handleEnded() {
      setIsPlaying(
        false
      );
    }


    audio.addEventListener(
      "timeupdate",
      handleTimeUpdate
    );

    audio.addEventListener(
      "loadedmetadata",
      handleMetadata
    );

    audio.addEventListener(
      "ended",
      handleEnded
    );


    return () => {
      audio.removeEventListener(
        "timeupdate",
        handleTimeUpdate
      );

      audio.removeEventListener(
        "loadedmetadata",
        handleMetadata
      );

      audio.removeEventListener(
        "ended",
        handleEnded
      );
    };
  }, [
    audioTimeline,
  ]);


  function togglePlayback() {
    const audio =
      audioRef.current;

    if (!audio) {
      return;
    }

    if (audio.paused) {
      audio
        .play()
        .then(
          () =>
            setIsPlaying(
              true
            )
        )
        .catch(
          () =>
            setIsPlaying(
              false
            )
        );
    } else {
      audio.pause();

      setIsPlaying(
        false
      );
    }
  }


  function seekTo(seconds) {
    const audio =
      audioRef.current;

    if (!audio) {
      return;
    }

    const safeTime =
      clamp(
        seconds,
        0,
        duration
        || 0
      );

    audio.currentTime =
      safeTime;

    setCurrentTime(
      safeTime
    );
  }


  function segmentLeft(
    segment
  ) {
    if (!duration) {
      return 0;
    }

    return (
      Number(
        segment
          .start_seconds
        || 0
      )
      / duration
    ) * 100;
  }


  function segmentWidth(
    segment
  ) {
    if (!duration) {
      return 0;
    }

    const start =
      Number(
        segment
          .start_seconds
        || 0
      );

    const end =
      Number(
        segment
          .end_seconds
        || 0
      );

    return (
      (
        end - start
      )
      / duration
    ) * 100;
  }


  function findSegmentAtTime(
    time
  ) {
    return segments.find(
      (segment) =>
        time >=
          Number(
            segment
              .start_seconds
            || 0
          )
        &&
        time <=
          Number(
            segment
              .end_seconds
            || 0
          )
    );
  }


  function preserveSegment(
    segment
  ) {
    if (!segment) {
      return;
    }

    setSelectedSegment(
      segment
    );

    setInvestigationLog(
      (current) => {
        const exists =
          current.some(
            (item) =>
              item.id
              === segment.id
          );

        if (exists) {
          return current;
        }

        return [
          segment,
          ...current,
        ].slice(
          0,
          12
        );
      }
    );
  }


  function selectSegment(
    segment
  ) {
    setActiveSegment(
      segment
    );

    preserveSegment(
      segment
    );

    seekTo(
      Number(
        segment
          .start_seconds
        || 0
      )
    );
  }


  function handleTimelineMove(
    event
  ) {
    const timeline =
      timelineRef.current;

    if (
      !timeline
      || !duration
    ) {
      return;
    }

    const bounds =
      timeline
        .getBoundingClientRect();

    const localX =
      clamp(
        event.clientX
        - bounds.left,
        0,
        bounds.width
      );

    const time =
      (
        localX
        / bounds.width
      ) * duration;

    const segment =
      findSegmentAtTime(
        time
      );

    setActiveSegment(
      segment
      || null
    );

    setTooltipPosition({
      x: clamp(
        localX + 20,
        12,
        Math.max(
          12,
          bounds.width - 370
        )
      ),

      y: 18,
    });
  }


  function handleTimelineClick(
    event
  ) {
    const timeline =
      timelineRef.current;

    if (
      !timeline
      || !duration
    ) {
      return;
    }

    const bounds =
      timeline
        .getBoundingClientRect();

    const localX =
      clamp(
        event.clientX
        - bounds.left,
        0,
        bounds.width
      );

    const time =
      (
        localX
        / bounds.width
      ) * duration;

    seekTo(
      time
    );

    const segment =
      findSegmentAtTime(
        time
      );

    preserveSegment(
      segment
    );
  }


  function removeLogItem(
    segmentId
  ) {
    setInvestigationLog(
      (current) =>
        current.filter(
          (item) =>
            item.id
            !== segmentId
        )
    );

    if (
      selectedSegment?.id
      === segmentId
    ) {
      setSelectedSegment(
        null
      );
    }
  }


  const displayedSegment =
    activeSegment
    || selectedSegment
    || findSegmentAtTime(
      currentTime
    );


  const playbackPercent =
    duration > 0
      ? clamp(
          (
            currentTime
            / duration
          ) * 100,
          0,
          100
        )
      : 0;
      async function handleAudioFeedback(verifiedLabel) {
  try {
    const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
    
    // Fallback prediction if summary isn't available
    const predictedLabel = audioSummary?.prediction || "Fake";

    const response = await fetch(`${API_URL}/feedback/audio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_path: audioUrl || "",
        audio_name: audioUrl ? audioUrl.split("/").pop() : "sample.wav",
        predicted_label: predictedLabel,
        verified_label: verifiedLabel,
        confidence: audioSummary?.confidence || 90.0,
        model_version: "v1",
      }),
    });

    const data = await response.json();

    if (response.ok && data.status === "success") {
      alert(`Feedback saved successfully! Logged as: ${verifiedLabel}`);
    } else {
      alert("Failed to save feedback: " + (data.message || "Unknown error"));
    }
  } catch (err) {
    console.error("Error submitting audio feedback:", err);
    alert("Unable to connect to feedback server.");
  }
}
      async function handleAudioVerifyLater(e) {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:8000/verify-later", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modality: "audio",
          identifier: audioUrl || "Audio_Track",
          prediction: audioSummary?.prediction || "Fake",
          confidence: audioSummary?.confidence || 0.8,
          filepath: audioUrl,
        }),
      });

      if (response.ok) {
        alert("Audio sample moved to Pending Verification!");
      } else {
        alert("Failed to move item to pending verification.");
      }
    } catch (err) {
      console.error("Error staging audio for verification:", err);
    }
  }


  return (
    <section className="audio-investigator">

      <div className="audio-investigator-header">

        <div>
          <p className="audio-kicker">
            FORGE AUDIO INVESTIGATION 3.2
          </p>

          <h2>
            Advanced Audio Forensic Workstation
          </h2>

          <p>
            Analyse synthetic speech through segment-level risk,
            acoustic curves, Voice DNA, pitch stability, energy,
            pauses, breathing behaviour and spectral evidence.
          </p>
        </div>


        <div className="audio-status-strip">

          <span>
            {segments.length}
            {" "}
            segments
          </span>

          <span>
            {
              suspiciousIntervals
                .length
            }
            {" "}
            suspicious intervals
          </span>

          <span>
            {
              normalizedPauseIntervals
                .length
            }
            {" "}
            pauses
          </span>

          <span>
            {
              normalizedBreathingEvents
                .length
            }
            {" "}
            breathing events
          </span>

          <span>
            {formatTime(
              duration
            )}
          </span>

        </div>

      </div>


      <div className="audio-player-console">

        <audio
          ref={audioRef}
          src={audioUrl}
          preload="metadata"
        />


        <button
          type="button"
          className="audio-play-button"
          onClick={
            togglePlayback
          }
        >
          {
            isPlaying
              ? "❚❚"
              : "▶"
          }
        </button>


        <div className="audio-time-readout">

          <strong>
            {formatTime(
              currentTime
            )}
          </strong>

          <span>/</span>

          <strong>
            {formatTime(
              duration
            )}
          </strong>

        </div>


        <div className="audio-player-track">

          <div
            className="audio-player-progress"
            style={{
              width:
                `${playbackPercent}%`,
            }}
          />

          <input
            type="range"
            min="0"
            max={
              duration || 0
            }
            step="0.01"
            value={
              currentTime
            }
            onChange={(
              event
            ) =>
              seekTo(
                Number(
                  event
                    .target
                    .value
                )
              )
            }
          />

        </div>

      </div>


      <VoiceDNAPanel
        voiceDNA={
          voiceDNA
        }
      />


      <AudioSummaryPanel
        summary={
          audioSummary
        }
      />


      <div className="audio-investigator-grid">

        <div className="audio-main-panel">

          <section className="audio-timeline-card">

            <div className="audio-section-title">

              <div>
                <p>
                  SEGMENT RISK TIMELINE
                </p>

                <h3>
                  Synthetic Speech Evidence
                </h3>
              </div>


              <div className="audio-risk-legend">

                <span className="risk-low">
                  Natural
                </span>

                <span className="risk-medium">
                  Suspicious
                </span>

                <span className="risk-high">
                  High AI Risk
                </span>

              </div>

            </div>


            <div
              ref={timelineRef}
              className="audio-risk-timeline"
              onMouseMove={
                handleTimelineMove
              }
              onMouseLeave={() =>
                setActiveSegment(
                  null
                )
              }
              onClick={
                handleTimelineClick
              }
            >

              <div
                className="audio-playhead"
                style={{
                  left:
                    `${playbackPercent}%`,
                }}
              />


              {segments.map(
                (segment) => (
                  <div
                    key={
                      segment.id
                    }
                    className={[
                      "audio-risk-segment",
                      riskClass(
                        segment
                          .risk_level
                      ),
                      selectedSegment?.id
                      === segment.id
                        ? "selected"
                        : "",
                    ].join(" ")}
                    style={{
                      left:
                        `${segmentLeft(
                          segment
                        )}%`,

                      width:
                        `${Math.max(
                          segmentWidth(
                            segment
                          ),
                          0.4
                        )}%`,
                    }}
                  />
                )
              )}


              {normalizedPauseIntervals.map(
                (
                  interval,
                  index
                ) => {
                  const left =
                    duration > 0
                      ? (
                          Number(
                            interval
                              .start_seconds
                            || 0
                          )
                          / duration
                        ) * 100
                      : 0;

                  const width =
                    duration > 0
                      ? (
                          (
                            Number(
                              interval
                                .end_seconds
                              || 0
                            )
                            -
                            Number(
                              interval
                                .start_seconds
                              || 0
                            )
                          )
                          / duration
                        ) * 100
                      : 0;

                  return (
                    <div
                      key={
                        `pause-${index}`
                      }
                      className="audio-pause-marker"
                      style={{
                        left:
                          `${left}%`,

                        width:
                          `${Math.max(
                            width,
                            0.25
                          )}%`,
                      }}
                      title={
                        `Pause ${interval.start} → ${interval.end}`
                      }
                    />
                  );
                }
              )}


              {normalizedBreathingEvents.map(
                (
                  event,
                  index
                ) => {
                  const left =
                    duration > 0
                      ? (
                          Number(
                            event
                              .time_seconds
                            || 0
                          )
                          / duration
                        ) * 100
                      : 0;

                  return (
                    <button
                      type="button"
                      key={
                        event.id
                        || index
                      }
                      className="audio-breath-marker"
                      style={{
                        left:
                          `${left}%`,
                      }}
                      onClick={(
                        clickEvent
                      ) => {
                        clickEvent
                          .stopPropagation();

                        seekTo(
                          Number(
                            event
                              .time_seconds
                            || 0
                          )
                        );
                      }}
                      title={
                        `Breathing candidate at ${event.time}`
                      }
                    >
                      B
                    </button>
                  );
                }
              )}


              {activeSegment && (
                <div
                  className="audio-floating-tooltip"
                  style={{
                    left:
                      tooltipPosition.x,

                    top:
                      tooltipPosition.y,
                  }}
                >

                  <div className="audio-tooltip-header">

                    <div>
                      <p>
                        SEGMENT ANALYSIS
                      </p>

                      <h3>
                        {
                          activeSegment.start
                        }
                        {" → "}
                        {
                          activeSegment.end
                        }
                      </h3>
                    </div>

                    <span
                      className={
                        riskClass(
                          activeSegment
                            .risk_level
                        )
                      }
                    >
                      {
                        activeSegment
                          .risk_level
                      }
                    </span>

                  </div>


                  <div className="audio-tooltip-score">

                    <span>
                      AI suspicion
                    </span>

                    <strong>
                      {
                        formatNumber(
                          activeSegment
                            .risk_score
                        )
                      }%
                    </strong>

                  </div>


                  <div className="audio-tooltip-verdict">

                    <span>
                      Prediction
                    </span>

                    <strong>
                      {
                        activeSegment
                          .prediction
                      }
                    </strong>

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {
                        formatNumber(
                          activeSegment
                            .confidence
                        )
                      }%
                    </strong>

                  </div>


                  <div className="audio-tooltip-metrics">

                    {Object.entries(
                      activeSegment
                        .metrics
                      || {}
                    )
                      .slice(
                        0,
                        7
                      )
                      .map(
                        ([
                          metric,
                          score,
                        ]) => (
                          <div
                            key={
                              metric
                            }
                          >

                            <section>

                              <span>
                                {
                                  SEGMENT_METRIC_LABELS[
                                    metric
                                  ]
                                  || metric
                                }
                              </span>

                              <b>
                                {
                                  formatNumber(
                                    score,
                                    1
                                  )
                                }%
                              </b>

                            </section>

                            <div>
                              <i
                                style={{
                                  width:
                                    `${clamp(
                                      score,
                                      0,
                                      100
                                    )}%`,
                                }}
                              />
                            </div>

                          </div>
                        )
                      )}

                  </div>


                  <ul>

                    {(
                      activeSegment
                        .reasons
                      || []
                    )
                      .slice(
                        0,
                        4
                      )
                      .map(
                        (reason) => (
                          <li
                            key={
                              reason
                            }
                          >
                            {reason}
                          </li>
                        )
                      )}

                  </ul>


                  <p className="audio-tooltip-hint">
                    Click to seek and preserve this segment.
                  </p>

                </div>
              )}

            </div>


            <div className="audio-time-scale">

              <span>
                00:00
              </span>

              <span>
                {
                  formatTime(
                    duration
                    * 0.25
                  )
                }
              </span>

              <span>
                {
                  formatTime(
                    duration
                    * 0.5
                  )
                }
              </span>

              <span>
                {
                  formatTime(
                    duration
                    * 0.75
                  )
                }
              </span>

              <span>
                {
                  formatTime(
                    duration
                  )
                }
              </span>

            </div>

            <div className="audio-timeline-explainer">
              <div className="timeline-explain-item natural">
                <span />
                <div>
                  <strong>Natural / Low risk</strong>
                  <p>Acoustic behaviour is close to the human-speech baseline.</p>
                </div>
              </div>

              <div className="timeline-explain-item suspicious">
                <span />
                <div>
                  <strong>Suspicious / Medium risk</strong>
                  <p>One or more parameters require focused manual review.</p>
                </div>
              </div>

              <div className="timeline-explain-item high">
                <span />
                <div>
                  <strong>High AI risk</strong>
                  <p>Multiple synthetic-speech indicators are present in the interval.</p>
                </div>
              </div>

              <div className="timeline-explain-item pause">
                <span />
                <div>
                  <strong>Amber markers</strong>
                  <p>Detected silence or pause intervals.</p>
                </div>
              </div>

              <div className="timeline-explain-item breath">
                <span>B</span>
                <div>
                  <strong>Breathing candidates</strong>
                  <p>Potential inhalation or exhalation events detected from low-energy noise.</p>
                </div>
              </div>
            </div>

          </section>


          <div className="audio-curve-grid">

            <CurveChart
              title="Pitch Contour"
              subtitle="Fundamental-frequency movement across time"
              data={
                audioCurves
                  ?.pitch
                || []
              }
              currentTime={
                currentTime
              }
              duration={
                duration
              }
              unit="Hz"
              onSeek={
                seekTo
              }
            />


            <CurveChart
              title="Energy Timeline"
              subtitle="Frame-level RMS energy and loudness variation"
              data={
                audioCurves
                  ?.energy
                || []
              }
              currentTime={
                currentTime
              }
              duration={
                duration
              }
              unit="RMS"
              onSeek={
                seekTo
              }
            />


            <CurveChart
              title="Spectral Flux"
              subtitle="Frame-to-frame spectral movement"
              data={
                audioCurves
                  ?.spectral_flux
                || []
              }
              currentTime={
                currentTime
              }
              duration={
                duration
              }
              unit="Flux"
              onSeek={
                seekTo
              }
            />


            <CurveChart
              title="Spectral Flatness"
              subtitle="Noise-like versus harmonic spectral behaviour"
              data={
                audioCurves
                  ?.spectral_flatness
                || []
              }
              currentTime={
                currentTime
              }
              duration={
                duration
              }
              unit="Ratio"
              onSeek={
                seekTo
              }
            />

          </div>


          <section className="audio-visual-grid">

            {waveformUrl && (
              <div className="audio-visual-card">

                <h3>
                  Waveform
                </h3>

                <img
                  src={
                    waveformUrl
                  }
                  alt="Audio waveform"
                />

                <div className="audio-visual-explainer">
                  <strong>How to read it</strong>
                  <p>Horizontal position represents time. Taller peaks indicate greater amplitude. Abrupt cuts, repeated envelopes, or unnaturally uniform sections may indicate editing or synthesis.</p>
                </div>

              </div>
            )}


            {spectrogramUrl && (
              <div className="audio-visual-card">

                <h3>
                  Spectrogram
                </h3>

                <img
                  src={
                    spectrogramUrl
                  }
                  alt="Audio spectrogram"
                />

                <div className="audio-visual-explainer">
                  <strong>How to read it</strong>
                  <p>Time runs left to right and frequency runs bottom to top. Brighter bands represent stronger energy. Repeated harmonic bands, missing transitions, and sharp spectral seams can support a synthetic-audio finding.</p>
                </div>

              </div>
            )}


            {heatmapUrl && (
              <div className="audio-visual-card">

                <h3>
                  Acoustic Heatmap
                </h3>

                <img
                  src={
                    heatmapUrl
                  }
                  alt="Audio forensic heatmap"
                />

                <div className="audio-heatmap-legend">
                  <div className="audio-heatmap-gradient" />
                  <div className="audio-heatmap-scale">
                    <span>Low anomaly</span>
                    <span>Needs review</span>
                    <span>High synthetic risk</span>
                  </div>
                </div>

                <div className="audio-visual-explainer">
                  <strong>Heatmap interpretation</strong>
                  <p>Cool colours indicate lower local anomaly. Green and yellow indicate moderate deviation. Orange and red indicate stronger synthetic or manipulation evidence. The heatmap is an explainability aid and must be interpreted with the model probability and acoustic parameters.</p>
                </div>

              </div>
            )}

          </section>

        </div>


        <aside className="audio-side-panel">

          <div className="audio-side-title">

            <p>
              LIVE SEGMENT ANALYSIS
            </p>

            <h3>
              {
                displayedSegment
                  ? `${displayedSegment.start} → ${displayedSegment.end}`
                  : "Move over timeline"
              }
            </h3>

          </div>


          {displayedSegment ? (
            <>

              <div className="audio-side-score">

                <span>
                  Segment AI suspicion
                </span>

                <strong>
                  {
                    formatNumber(
                      displayedSegment
                        .risk_score
                    )
                  }%
                </strong>

                <div>
                  <i
                    style={{
                      width:
                        `${clamp(
                          displayedSegment
                            .risk_score,
                          0,
                          100
                        )}%`,
                    }}
                  />
                </div>

              </div>


              <div className="audio-side-summary">

                <div>
                  <span>
                    Prediction
                  </span>

                  <strong>
                    {
                      displayedSegment
                        .prediction
                    }
                  </strong>
                </div>

                <div>
                  <span>
                    Confidence
                  </span>

                  <strong>
                    {
                      formatNumber(
                        displayedSegment
                          .confidence
                      )
                    }%
                  </strong>
                </div>

                <div>
                  <span>
                    Risk
                  </span>

                  <strong>
                    {
                      displayedSegment
                        .risk_level
                    }
                  </strong>
                </div>

              </div>


              <div className="audio-metric-list">

                {Object.entries(
                  displayedSegment
                    .metrics
                  || {}
                ).map(
                  ([
                    metric,
                    score,
                  ]) => (
                    <div
                      key={
                        metric
                      }
                    >

                      <section>

                        <span>
                          {
                            SEGMENT_METRIC_LABELS[
                              metric
                            ]
                            || metric
                          }
                        </span>

                        <strong>
                          {
                            formatNumber(
                              score,
                              1
                            )
                          }%
                        </strong>

                      </section>

                      <div>
                        <i
                          style={{
                            width:
                              `${clamp(
                                score,
                                0,
                                100
                              )}%`,
                          }}
                        />
                      </div>

                    </div>
                  )
                )}

              </div>


              <div className="audio-findings">

                <h4>
                  Segment Findings
                </h4>

                {(
                  displayedSegment
                    .reasons
                  || []
                ).map(
                  (reason) => (
                    <p
                      key={
                        reason
                      }
                    >
                      {reason}
                    </p>
                  )
                )}

              </div>

            </>
          ) : (
            <div className="audio-side-empty">

              <div className="audio-radar">
                <span />
                <i />
              </div>

              <p>
                Move across the timeline to inspect
                pitch, phase, noise, energy, pauses
                and frequency evidence.
              </p>

            </div>
          )}


          <div className="audio-ranked-segments">

            <h4>
              Most Suspicious Segments
            </h4>

            {rankedSegments
              .slice(
                0,
                8
              )
              .map(
                (
                  segment,
                  index
                ) => (
                  <button
                    type="button"
                    key={
                      segment.id
                    }
                    className={
                      selectedSegment?.id
                      === segment.id
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      selectSegment(
                        segment
                      )
                    }
                  >

                    <span>
                      {
                        String(
                          index + 1
                        ).padStart(
                          2,
                          "0"
                        )
                      }
                    </span>

                    <div>
                      <b>
                        {
                          segment.start
                        }
                        {" → "}
                        {
                          segment.end
                        }
                      </b>

                      <small>
                        {
                          segment
                            .risk_level
                        }
                      </small>
                    </div>

                    <strong>
                      {
                        formatNumber(
                          segment
                            .risk_score,
                          1
                        )
                      }%
                    </strong>

                  </button>
                )
              )}

          </div>
          {/* CONTINUOUS LEARNING FEEDBACK BUTTONS */}
<div style={{ marginTop: "20px", paddingTop: "15px", borderTop: "1px solid rgba(255, 255, 255, 0.08)" }}>
  <h4>Continuous Learning Feedback</h4>
  <div style={{ display: "flex", gap: "6px", marginTop: "10px" }}>
    <button
      type="button"
      onClick={() => handleAudioFeedback("Real")}
      style={{ flex: 1, padding: "8px", borderRadius: "8px", background: "#00e676", color: "#000", fontWeight: "bold", border: "none", cursor: "pointer" }}
    >
      ✓ Correct
    </button>
    <button
      type="button"
      onClick={() => handleAudioFeedback("Fake")}
      style={{ flex: 1, padding: "8px", borderRadius: "8px", background: "#ff3d71", color: "#fff", fontWeight: "bold", border: "none", cursor: "pointer" }}
    >
      ✕ Incorrect
    </button>
    <button
      type="button"
      onClick={handleAudioVerifyLater}
      style={{ flex: 1, padding: "8px", borderRadius: "8px", background: "#ffd166", color: "#000", fontWeight: "bold", border: "none", cursor: "pointer" }}
    >
      ⏱ Verify Later
    </button>
  </div>
</div>

        </aside>

      </div>


      <div className="audio-bottom-grid">

        <section className="audio-log-panel">

          <div className="audio-log-header">

            <div>
              <p>
                INVESTIGATION LOG
              </p>

              <h3>
                Preserved Audio Evidence
              </h3>
            </div>

            <span>
              {
                investigationLog
                  .length
              }
              {" "}
              entries
            </span>

          </div>


          {investigationLog.length ? (
            <div className="audio-log-list">

              {investigationLog.map(
                (
                  segment,
                  index
                ) => (
                  <div
                    className="audio-log-item"
                    key={
                      segment.id
                    }
                  >

                    <button
                      type="button"
                      onClick={() =>
                        selectSegment(
                          segment
                        )
                      }
                    >

                      <span>
                        {
                          String(
                            index + 1
                          ).padStart(
                            2,
                            "0"
                          )
                        }
                      </span>

                      <div>
                        <b>
                          {
                            segment.start
                          }
                          {" → "}
                          {
                            segment.end
                          }
                        </b>

                        <small>
                          {
                            segment
                              .prediction
                          }
                          {" • "}
                          {
                            segment
                              .risk_level
                          }
                        </small>
                      </div>

                      <strong>
                        {
                          formatNumber(
                            segment
                              .risk_score,
                            1
                          )
                        }%
                      </strong>

                    </button>


                    <button
                      type="button"
                      className="audio-log-remove"
                      onClick={() =>
                        removeLogItem(
                          segment.id
                        )
                      }
                    >
                      ×
                    </button>

                  </div>
                )
              )}

            </div>
          ) : (
            <p className="audio-log-empty">
              Click suspicious timeline segments to preserve them here.
            </p>
          )}

        </section>


        <section className="audio-interval-panel">

          <div className="audio-log-header">

            <div>
              <p>
                SUSPICIOUS INTERVALS
              </p>

              <h3>
                Merged Forensic Events
              </h3>
            </div>

            <span>
              {
                suspiciousIntervals
                  .length
              }
              {" "}
              intervals
            </span>

          </div>


          <div className="audio-interval-list">

            {suspiciousIntervals.map(
              (
                interval,
                index
              ) => (
                <button
                  type="button"
                  key={
                    interval.id
                    || index
                  }
                  className={
                    riskClass(
                      interval
                        .risk_level
                      || interval
                        .risk
                    )
                  }
                  onClick={() =>
                    seekTo(
                      Number(
                        interval
                          .start_seconds
                        || 0
                      )
                    )
                  }
                >

                  <span>
                    {
                      String(
                        index + 1
                      ).padStart(
                        2,
                        "0"
                      )
                    }
                  </span>

                  <div>
                    <b>
                      {
                        interval.start
                      }
                      {" → "}
                      {
                        interval.end
                      }
                    </b>

                    <small>
                      {
                        interval.reason
                        || "Synthetic acoustic evidence detected."
                      }
                    </small>
                  </div>

                  <strong>
                    {
                      formatNumber(
                        interval
                          .risk_score
                        || interval
                          .score,
                        1
                      )
                    }%
                  </strong>

                </button>
              )
            )}

          </div>

        </section>

      </div>


      <p className="audio-investigator-disclaimer">
        Segment scores, Voice DNA values, breathing markers and acoustic
        curves are explainable forensic indicators. They should be reviewed
        alongside the CNN-BiLSTM prediction and should not be treated as
        independent proof of manipulation.
      </p>

    </section>
  );
}


export default InteractiveAudioInvestigator;