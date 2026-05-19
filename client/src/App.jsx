import { useState, useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";

const CLUSTER_COLORS = ["#7F77DD", "#1D9E75", "#D85A30", "#D4537E", "#378ADD", "#639922", "#BA7517"];
const CLUSTER_LABELS = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6"];

const YEARS = d3.range(1960, 2011);

function generateThemeRiverData() {
  const seeds = [
    [40, 5, 8, 12, 3, 6, 4],
    [38, 8, 10, 14, 5, 8, 6],
    [35, 12, 12, 16, 7, 9, 7],
    [30, 18, 15, 16, 10, 11, 8],
    [25, 22, 18, 14, 14, 13, 10],
    [22, 26, 20, 12, 16, 14, 12],
    [20, 28, 22, 11, 18, 15, 13],
    [18, 26, 24, 12, 20, 17, 14],
    [15, 24, 26, 14, 22, 18, 16],
    [14, 22, 24, 16, 24, 20, 18],
  ];
  return YEARS.map((year, i) => {
    const decade = Math.floor(i / 5);
    const base = seeds[Math.min(decade, seeds.length - 1)];
    const row = { year };
    CLUSTER_LABELS.forEach((_, ci) => {
      row[`cluster_${ci}`] = Math.max(1, base[ci] + Math.round((Math.random() - 0.5) * 4));
    });
    return row;
  });
}

function generateUMAPData() {
  const points = [];
  const decadeColors = d3.scaleSequential(d3.interpolateCool).domain([1960, 2010]);
  for (let i = 0; i < 300; i++) {
    const year = 1960 + Math.floor(Math.random() * 51);
    const cluster = Math.floor(Math.random() * 7);
    const angle = (cluster / 7) * Math.PI * 2;
    const r = 3 + Math.random() * 3;
    const drift = (year - 1960) / 50;
    points.push({
      x: Math.cos(angle) * r + (Math.random() - 0.5) * 3 + drift * 2,
      y: Math.sin(angle) * r + (Math.random() - 0.5) * 3 + drift * 1.5,
      year,
      cluster,
    });
  }
  return points;
}

function generateFeatureData(clusterId) {
  const profiles = [
    { energy: 0.72, danceability: 0.65, loudness: 0.68, acousticness: 0.22, valence: 0.60, tempo: 0.58 },
    { energy: 0.55, danceability: 0.80, loudness: 0.55, acousticness: 0.30, valence: 0.75, tempo: 0.72 },
    { energy: 0.85, danceability: 0.45, loudness: 0.82, acousticness: 0.10, valence: 0.40, tempo: 0.65 },
    { energy: 0.40, danceability: 0.55, loudness: 0.35, acousticness: 0.75, valence: 0.55, tempo: 0.42 },
    { energy: 0.68, danceability: 0.70, loudness: 0.62, acousticness: 0.20, valence: 0.68, tempo: 0.80 },
    { energy: 0.60, danceability: 0.60, loudness: 0.58, acousticness: 0.45, valence: 0.50, tempo: 0.55 },
    { energy: 0.78, danceability: 0.72, loudness: 0.75, acousticness: 0.15, valence: 0.62, tempo: 0.70 },
  ];
  return profiles[clusterId % profiles.length];
}

function generateWordCloud(clusterId) {
  const clouds = [
    [{ word: "love", w: 0.92 }, { word: "heart", w: 0.80 }, { word: "baby", w: 0.75 }, { word: "night", w: 0.68 }, { word: "feel", w: 0.60 }, { word: "soul", w: 0.55 }, { word: "dream", w: 0.50 }, { word: "dance", w: 0.45 }, { word: "sweet", w: 0.40 }, { word: "time", w: 0.35 }, { word: "gone", w: 0.30 }, { word: "rain", w: 0.28 }],
    [{ word: "yeah", w: 0.88 }, { word: "gonna", w: 0.78 }, { word: "rock", w: 0.72 }, { word: "roll", w: 0.65 }, { word: "free", w: 0.60 }, { word: "road", w: 0.52 }, { word: "wild", w: 0.48 }, { word: "fire", w: 0.44 }, { word: "ride", w: 0.38 }, { word: "hard", w: 0.33 }, { word: "loud", w: 0.28 }, { word: "burn", w: 0.24 }],
    [{ word: "street", w: 0.90 }, { word: "money", w: 0.82 }, { word: "real", w: 0.74 }, { word: "life", w: 0.68 }, { word: "world", w: 0.61 }, { word: "game", w: 0.55 }, { word: "power", w: 0.49 }, { word: "hustle", w: 0.44 }, { word: "grind", w: 0.38 }, { word: "flow", w: 0.32 }, { word: "rise", w: 0.27 }, { word: "truth", w: 0.22 }],
    [{ word: "blue", w: 0.85 }, { word: "gone", w: 0.76 }, { word: "lonesome", w: 0.70 }, { word: "cry", w: 0.63 }, { word: "miss", w: 0.57 }, { word: "pain", w: 0.51 }, { word: "tears", w: 0.45 }, { word: "home", w: 0.40 }, { word: "alone", w: 0.35 }, { word: "cold", w: 0.30 }, { word: "dark", w: 0.26 }, { word: "hurt", w: 0.22 }],
    [{ word: "dance", w: 0.91 }, { word: "groove", w: 0.83 }, { word: "beat", w: 0.76 }, { word: "move", w: 0.69 }, { word: "feel", w: 0.62 }, { word: "rhythm", w: 0.55 }, { word: "music", w: 0.50 }, { word: "floor", w: 0.44 }, { word: "night", w: 0.38 }, { word: "bass", w: 0.33 }, { word: "party", w: 0.28 }, { word: "body", w: 0.24 }],
    [{ word: "peace", w: 0.87 }, { word: "soul", w: 0.78 }, { word: "earth", w: 0.71 }, { word: "spirit", w: 0.64 }, { word: "grace", w: 0.58 }, { word: "light", w: 0.51 }, { word: "lord", w: 0.46 }, { word: "bless", w: 0.40 }, { word: "holy", w: 0.35 }, { word: "pray", w: 0.30 }, { word: "faith", w: 0.25 }, { word: "joy", w: 0.21 }],
    [{ word: "stars", w: 0.89 }, { word: "sky", w: 0.79 }, { word: "fly", w: 0.72 }, { word: "high", w: 0.65 }, { word: "dream", w: 0.58 }, { word: "shine", w: 0.52 }, { word: "bright", w: 0.46 }, { word: "wonder", w: 0.41 }, { word: "soar", w: 0.36 }, { word: "glow", w: 0.31 }, { word: "rise", w: 0.26 }, { word: "above", w: 0.21 }],
  ];
  return clouds[clusterId % clouds.length];
}

// ─── ThemeRiver ──────────────────────────────────────────────────────────────
function ThemeRiver({ data, activeCluster, onYearHover }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    const el = svgRef.current;
    const W = el.clientWidth || 620;
    const H = 220;
    d3.select(el).selectAll("*").remove();

    const svg = d3.select(el)
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("width", "100%")
      .attr("height", H);

    const keys = CLUSTER_LABELS.map((_, i) => `cluster_${i}`);
    const stack = d3.stack().keys(keys).offset(d3.stackOffsetWiggle).order(d3.stackOrderInsideOut);
    const series = stack(data);

    const xScale = d3.scaleLinear()
      .domain([data[0].year, data[data.length - 1].year])
      .range([40, W - 20]);

    const yExtent = [
      d3.min(series, s => d3.min(s, d => d[0])),
      d3.max(series, s => d3.max(s, d => d[1])),
    ];
    const yScale = d3.scaleLinear().domain(yExtent).range([H - 20, 20]);

    const area = d3.area()
      .x(d => xScale(d.data.year))
      .y0(d => yScale(d[0]))
      .y1(d => yScale(d[1]))
      .curve(d3.curveBasis);

    svg.append("g")
      .selectAll("path")
      .data(series)
      .join("path")
      .attr("d", area)
      .attr("fill", (_, i) => CLUSTER_COLORS[i])
      .attr("opacity", (_, i) => {
        if (activeCluster === null) return 0.82;
        return i === activeCluster ? 1 : 0.18;
      })
      .style("cursor", "crosshair")
      .on("mousemove", function (event, d) {
        const [mx] = d3.pointer(event, el);
        const year = Math.round(xScale.invert(mx));
        const row = data.find(r => r.year === year);
        if (row) {
          const ci = parseInt(d.key.split("_")[1]);
          setTooltip({ x: mx, y: event.offsetY, year, cluster: ci, value: row[d.key] });
          onYearHover && onYearHover(year);
        }
      })
      .on("mouseleave", () => { setTooltip(null); onYearHover && onYearHover(null); });

    const xAxis = d3.axisBottom(xScale)
      .ticks(10)
      .tickFormat(d3.format("d"))
      .tickSize(0);
    svg.append("g")
      .attr("transform", `translate(0,${H - 18})`)
      .call(xAxis)
      .call(g => g.select(".domain").remove())
      .selectAll("text")
      .style("font-size", "11px")
      .style("fill", "var(--color-text-secondary)");
  }, [data, activeCluster]);

  return (
    <div style={{ position: "relative" }}>
      <svg ref={svgRef} style={{ display: "block", width: "100%" }} />
      {tooltip && (
        <div style={{
          position: "absolute",
          left: tooltip.x + 12,
          top: tooltip.y - 10,
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderRadius: "var(--border-radius-md)",
          padding: "8px 12px",
          fontSize: 12,
          pointerEvents: "none",
          zIndex: 10,
          boxShadow: "none",
        }}>
          <div style={{ fontWeight: 500, marginBottom: 4 }}>{tooltip.year}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: CLUSTER_COLORS[tooltip.cluster], display: "inline-block" }} />
            <span style={{ color: "var(--color-text-secondary)" }}>{CLUSTER_LABELS[tooltip.cluster]}:</span>
            <span style={{ fontWeight: 500 }}>{tooltip.value} songs</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── UMAP Scatter ─────────────────────────────────────────────────────────────
function UMAPScatter({ points, activeCluster, onClusterClick }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !points.length) return;
    const el = svgRef.current;
    const W = el.clientWidth || 280;
    const H = 260;
    d3.select(el).selectAll("*").remove();

    const svg = d3.select(el)
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("width", "100%")
      .attr("height", H);

    const xExt = d3.extent(points, d => d.x);
    const yExt = d3.extent(points, d => d.y);
    const pad = 20;
    const xScale = d3.scaleLinear().domain(xExt).range([pad, W - pad]);
    const yScale = d3.scaleLinear().domain(yExt).range([H - pad, pad]);

    svg.selectAll("circle")
      .data(points)
      .join("circle")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 3.5)
      .attr("fill", d => CLUSTER_COLORS[d.cluster])
      .attr("opacity", d => {
        if (activeCluster === null) return 0.70;
        return d.cluster === activeCluster ? 0.95 : 0.12;
      })
      .style("cursor", "pointer")
      .on("click", (_, d) => onClusterClick && onClusterClick(d.cluster));

    svg.append("text")
      .attr("x", pad)
      .attr("y", H - 4)
      .style("font-size", "10px")
      .style("fill", "var(--color-text-secondary)")
      .text("UMAP dim 1 →");

    svg.append("text")
      .attr("transform", `rotate(-90)`)
      .attr("x", -H + pad)
      .attr("y", 11)
      .style("font-size", "10px")
      .style("fill", "var(--color-text-secondary)")
      .text("UMAP dim 2 →");
  }, [points, activeCluster]);

  return <svg ref={svgRef} style={{ display: "block", width: "100%" }} />;
}

// ─── Audio Feature Bars ───────────────────────────────────────────────────────
function AudioFeatureBar({ label, value, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <div style={{ width: 80, fontSize: 12, color: "var(--color-text-secondary)", textAlign: "right", flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1, height: 8, background: "var(--color-background-secondary)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${value * 100}%`,
          background: color,
          borderRadius: 4,
          transition: "width 0.5s ease",
        }} />
      </div>
      <div style={{ width: 32, fontSize: 12, fontWeight: 500, flexShrink: 0 }}>{value.toFixed(2)}</div>
    </div>
  );
}

// ─── Word Cloud ───────────────────────────────────────────────────────────────
function WordCloud({ words, color }) {
  const sorted = [...words].sort((a, b) => b.w - a.w);
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 8px", alignItems: "baseline", minHeight: 80 }}>
      {sorted.map(({ word, w }) => (
        <span
          key={word}
          style={{
            fontSize: `${Math.round(11 + w * 18)}px`,
            fontWeight: w > 0.7 ? 500 : 400,
            color: w > 0.65 ? color : "var(--color-text-secondary)",
            lineHeight: 1.3,
            transition: "all 0.4s ease",
          }}
        >
          {word}
        </span>
      ))}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeCluster, setActiveCluster] = useState(0);
  const [hoveredYear, setHoveredYear] = useState(null);


  const themeData = useMemo(() => generateThemeRiverData(), []);
  const umapPoints = useMemo(() => generateUMAPData(), []);
  const features = generateFeatureData(activeCluster);
  const words = generateWordCloud(activeCluster);

  const featureEntries = [
    { label: "Energy", key: "energy" },
    { label: "Danceability", key: "danceability" },
    { label: "Loudness", key: "loudness" },
    { label: "Acousticness", key: "acousticness" },
    { label: "Valence", key: "valence" },
    { label: "Tempo", key: "tempo" },
  ];

  const clusterColor = CLUSTER_COLORS[activeCluster];

  return (
    <div style={{
      fontFamily: "var(--font-sans)",
      background: "var(--color-background-tertiary)",
      minHeight: "100vh",
      padding: "0 0 2rem",
    }}>
      {/* Header */}
      <div style={{
        background: "var(--color-background-primary)",
        borderBottom: "0.5px solid var(--color-border-tertiary)",
        padding: "14px 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <h1 style={{ fontSize: 16, fontWeight: 500, margin: 0 }}>Multi-modal Music Trend Analysis</h1>
          <span style={{
            fontSize: 12,
            color: "var(--color-text-secondary)",
            background: "var(--color-background-secondary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-md)",
            padding: "2px 10px",
          }}>1960 – 2010</span>
        </div>
        <div style={{
          fontSize: 13,
          padding: "5px 14px",
          borderRadius: 99,
          border: `0.5px solid ${clusterColor}`,
          background: clusterColor + "18",
          color: clusterColor,
          fontWeight: 500,
        }}>
          Audio + Lyrics
        </div>
      </div>

      <div style={{ padding: "20px 24px", display: "grid", gridTemplateColumns: "1fr 320px", gap: 16 }}>

        {/* LEFT COLUMN */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* ThemeRiver Card */}
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "16px 20px",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                ThemeRiver — cluster trends, 1960–2010
              </div>
              {hoveredYear && (
                <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>year: {hoveredYear}</span>
              )}
            </div>

            {/* Legend */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", marginBottom: 12 }}>
              {CLUSTER_LABELS.map((label, i) => (
                <button
                  key={i}
                  onClick={() => setActiveCluster(activeCluster === i ? null : i)}
                  style={{
                    display: "flex", alignItems: "center", gap: 5,
                    fontSize: 12,
                    background: "none", border: "none", cursor: "pointer", padding: 0,
                    color: activeCluster === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                    fontWeight: activeCluster === i ? 500 : 400,
                    opacity: activeCluster !== null && activeCluster !== i ? 0.5 : 1,
                    transition: "all 0.2s",
                  }}
                >
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: CLUSTER_COLORS[i], flexShrink: 0 }} />
                  {label}
                </button>
              ))}
            </div>

            <ThemeRiver
              data={themeData}
              activeCluster={activeCluster}
              onYearHover={setHoveredYear}
            />
          </div>

          {/* Bottom row: Audio Features + Word Cloud */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

            {/* Audio Features */}
            <div style={{
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-lg)",
              padding: "16px 20px",
            }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 14 }}>
                Audio features — {activeCluster !== null ? CLUSTER_LABELS[activeCluster] : "all clusters"}
              </div>
              {featureEntries.map(({ label, key }) => (
                <AudioFeatureBar
                  key={key}
                  label={label}
                  value={features[key]}
                  color={clusterColor}
                />
              ))}
            </div>

            {/* Word Cloud */}
            <div style={{
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-lg)",
              padding: "16px 20px",
            }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 14 }}>
                Top lyric words — {activeCluster !== null ? CLUSTER_LABELS[activeCluster] : "all clusters"}
              </div>
              <WordCloud words={words} color={clusterColor} />
            </div>

          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Stats cards */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { label: "Songs", value: "~10K", sub: "1960–2010" },
              { label: "Modalities", value: "2", sub: "audio + lyrics" },
            ].map(({ label, value, sub }) => (
              <div key={label} style={{
                background: "var(--color-background-secondary)",
                borderRadius: "var(--border-radius-md)",
                padding: "12px 14px",
              }}>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
                <div style={{ fontSize: 24, fontWeight: 500, lineHeight: 1 }}>{value}</div>
                <div style={{ fontSize: 11, color: clusterColor, marginTop: 4 }}>{sub}</div>
              </div>
            ))}
          </div>

          {/* UMAP Card */}
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "16px 20px",
            flex: 1,
          }}>
            <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 4 }}>
              UMAP — song clusters
            </div>

            {/* Decade gradient legend */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
              <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>Earlier</span>
              <div style={{
                flex: 1,
                height: 6,
                borderRadius: 3,
                background: "linear-gradient(to right, #EEEDFE, #7F77DD, #1D9E75)",
              }} />
              <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>Later</span>
            </div>

            <UMAPScatter
              points={umapPoints}
              activeCluster={activeCluster}
              onClusterClick={i => setActiveCluster(activeCluster === i ? null : i)}
            />

            {/* Cluster swatches */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 10px", marginTop: 10 }}>
              {CLUSTER_LABELS.map((label, i) => (
                <button
                  key={i}
                  onClick={() => setActiveCluster(activeCluster === i ? null : i)}
                  style={{
                    display: "flex", alignItems: "center", gap: 4,
                    fontSize: 11,
                    background: "none", border: "none", cursor: "pointer", padding: 0,
                    color: activeCluster === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                    fontWeight: activeCluster === i ? 500 : 400,
                    opacity: activeCluster !== null && activeCluster !== i ? 0.45 : 1,
                    transition: "all 0.2s",
                  }}
                >
                  <span style={{ width: 6, height: 6, borderRadius: 99, background: CLUSTER_COLORS[i], display: "inline-block" }} />
                  {label}
                </button>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
