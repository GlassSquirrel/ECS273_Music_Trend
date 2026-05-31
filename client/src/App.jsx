import { useState, useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import * as THREE from "three";
import clusterPoints from "./clusterData.json";
import wordCloudData from "./wordCloudData.json";

const CLUSTER_COLORS = [
  "#7F77DD",
  "#1D9E75",
  "#D85A30",
  "#D4537E",
  "#378ADD",
  "#639922",
  "#BA7517",
  "#9966CC",
];

const CLUSTER_LABELS = [
  "Cluster 0",
  "Cluster 1",
  "Cluster 2",
  "Cluster 3",
  "Cluster 4",
  "Cluster 5",
  "Cluster 6",
  "Cluster 7",
];

const YEARS = d3.range(1960, 2011);

function generateThemeRiverData() {
  const seeds = [
    [40, 5, 8, 12, 3, 6, 4, 5],
    [38, 8, 10, 14, 5, 8, 6, 4],
    [35, 12, 12, 16, 7, 9, 7, 5],
    [30, 18, 15, 16, 10, 11, 8, 7],
    [25, 22, 18, 14, 14, 13, 10, 9],
    [22, 26, 20, 12, 16, 14, 12, 10],
    [20, 28, 22, 11, 18, 15, 13, 11],
    [18, 26, 24, 12, 20, 17, 14, 12],
    [15, 24, 26, 14, 22, 18, 16, 13],
    [14, 22, 24, 16, 24, 20, 18, 14],
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

function generateFeatureData(clusterId) {
  const profiles = [
    { energy: 0.72, danceability: 0.65, loudness: 0.68, acousticness: 0.22, valence: 0.60, tempo: 0.58 },
    { energy: 0.55, danceability: 0.80, loudness: 0.55, acousticness: 0.30, valence: 0.75, tempo: 0.72 },
    { energy: 0.85, danceability: 0.45, loudness: 0.82, acousticness: 0.10, valence: 0.40, tempo: 0.65 },
    { energy: 0.40, danceability: 0.55, loudness: 0.35, acousticness: 0.75, valence: 0.55, tempo: 0.42 },
    { energy: 0.68, danceability: 0.70, loudness: 0.62, acousticness: 0.20, valence: 0.68, tempo: 0.80 },
    { energy: 0.60, danceability: 0.60, loudness: 0.58, acousticness: 0.45, valence: 0.50, tempo: 0.55 },
    { energy: 0.78, danceability: 0.72, loudness: 0.75, acousticness: 0.15, valence: 0.62, tempo: 0.70 },
    { energy: 0.50, danceability: 0.68, loudness: 0.52, acousticness: 0.55, valence: 0.65, tempo: 0.60 },
  ];
  return profiles[clusterId % profiles.length];
}

function getWordCloud(clusterId) {
  return wordCloudData[String(clusterId)] ?? [];
}

// ─── Soft circle texture for 3D points ───────────────────────────────────────
function createPointTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0.0, "rgba(255,255,255,1.0)");
  gradient.addColorStop(0.4, "rgba(255,255,255,0.9)");
  gradient.addColorStop(0.8, "rgba(255,255,255,0.3)");
  gradient.addColorStop(1.0, "rgba(255,255,255,0.0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

// ─── 3D Cluster Scatter (Three.js) ───────────────────────────────────────────
function ClusterScatter3D({ points, activeCluster, onClusterClick }) {
  const containerRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [isDraggingCursor, setIsDraggingCursor] = useState(false);
  const activeClusterRef = useRef(activeCluster);
  const onClickRef = useRef(onClusterClick);

  useEffect(() => { activeClusterRef.current = activeCluster; }, [activeCluster]);
  useEffect(() => { onClickRef.current = onClusterClick; }, [onClusterClick]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !points.length) return;

    let mounted = true;
    const W = container.clientWidth || 300;
    const H = 340;

    // ── Scene & renderer ──────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0c0b);

    const camera = new THREE.PerspectiveCamera(52, W / H, 0.1, 200);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.display = "block";
    container.appendChild(renderer.domElement);

    // ── Normalize coordinates to [-5, 5] ─────────────────────────────────────
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const zs = points.map(p => p.z);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const zMin = Math.min(...zs), zMax = Math.max(...zs);
    const SPAN = 10;
    const norm = (v, mn, mx) => ((v - mn) / (mx - mn) - 0.5) * SPAN;

    const lookAt = new THREE.Vector3(0, 0, 0);

    // ── Group points by cluster ───────────────────────────────────────────────
    const clustersMap = {};
    for (const p of points) {
      if (!clustersMap[p.cluster]) clustersMap[p.cluster] = [];
      clustersMap[p.cluster].push(p);
    }

    // ── Shared circle texture ─────────────────────────────────────────────────
    const pointTex = createPointTexture();

    // ── Build one Points cloud per cluster ───────────────────────────────────
    const pointClouds = [];
    const clusterIds = Object.keys(clustersMap).map(Number).sort();

    for (const cid of clusterIds) {
      const pts = clustersMap[cid];
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(pts.length * 3);

      for (let i = 0; i < pts.length; i++) {
        pos[i * 3]     = norm(pts[i].x, xMin, xMax);
        pos[i * 3 + 1] = norm(pts[i].y, yMin, yMax);
        pos[i * 3 + 2] = norm(pts[i].z, zMin, zMax);
      }
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));

      const mat = new THREE.PointsMaterial({
        color: new THREE.Color(CLUSTER_COLORS[cid]),
        size: 0.28,
        map: pointTex,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
        alphaTest: 0.02,
      });

      const cloud = new THREE.Points(geo, mat);
      cloud.userData = { clusterId: cid, pts };
      scene.add(cloud);
      pointClouds.push(cloud);
    }

    // ── Subtle grid helper ────────────────────────────────────────────────────
    const grid = new THREE.GridHelper(SPAN, 10, 0x1a1a18, 0x1a1a18);
    grid.position.y = -SPAN / 2;
    grid.material.opacity = 0.5;
    grid.material.transparent = true;
    scene.add(grid);

    // ── Camera orbit state ────────────────────────────────────────────────────
    let theta = 0.6;
    let phi = 1.15;
    let radius = 20;
    let isDragging = false;
    let lastX = 0, lastY = 0;
    let autoRotate = true;
    let resumeTimer = null;

    const updateCamera = () => {
      camera.position.set(
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.cos(theta),
      );
      camera.lookAt(lookAt);
    };
    updateCamera();

    // ── Orbit & zoom handlers ─────────────────────────────────────────────────
    const pauseAutoRotate = () => {
      autoRotate = false;
      clearTimeout(resumeTimer);
      resumeTimer = setTimeout(() => { autoRotate = true; }, 3500);
    };

    const onMouseDown = (e) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      pauseAutoRotate();
      if (mounted) setIsDraggingCursor(true);
    };

    const onMouseMove = (e) => {
      if (!isDragging) return;
      theta -= (e.clientX - lastX) * 0.007;
      phi = Math.max(0.12, Math.min(Math.PI - 0.12, phi - (e.clientY - lastY) * 0.007));
      lastX = e.clientX;
      lastY = e.clientY;
      updateCamera();
      if (mounted) setTooltip(null);
    };

    const onMouseUp = () => {
      isDragging = false;
      if (mounted) setIsDraggingCursor(false);
    };

    const onWheel = (e) => {
      e.preventDefault();
      radius = Math.max(9, Math.min(38, radius + e.deltaY * 0.04));
      updateCamera();
      pauseAutoRotate();
    };

    // ── Raycasting for hover & click ──────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.35;
    const mouse = new THREE.Vector2();
    let hoverTick = 0;

    const onCanvasMouseMove = (e) => {
      if (isDragging) return;
      hoverTick++;
      if (hoverTick % 2 !== 0) return;

      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);

      const hits = raycaster.intersectObjects(pointClouds);
      if (hits.length > 0) {
        const obj = hits[0].object;
        const idx = hits[0].index;
        const pt = obj.userData.pts[idx];
        if (mounted) {
          setTooltip({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            title: pt.title || "—",
            artist: pt.artist || "",
            cluster: pt.cluster,
            year: pt.year > 0 ? pt.year : null,
          });
        }
      } else {
        if (mounted) setTooltip(null);
      }
    };

    const onCanvasClick = (e) => {
      if (isDragging) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(pointClouds);
      if (hits.length > 0) {
        const cid = hits[0].object.userData.clusterId;
        onClickRef.current && onClickRef.current(cid);
      }
    };

    const onMouseLeave = () => {
      if (mounted) setTooltip(null);
    };

    renderer.domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
    renderer.domElement.addEventListener("mousemove", onCanvasMouseMove);
    renderer.domElement.addEventListener("click", onCanvasClick);
    renderer.domElement.addEventListener("mouseleave", onMouseLeave);

    // ── Animation loop ────────────────────────────────────────────────────────
    let animFrame;
    const animate = () => {
      animFrame = requestAnimationFrame(animate);

      if (autoRotate) {
        theta += 0.0025;
        updateCamera();
      }

      const ac = activeClusterRef.current;
      for (const cloud of pointClouds) {
        const cid = cloud.userData.clusterId;
        const mat = cloud.material;
        if (ac === null) {
          mat.opacity = 0.78;
          mat.size = 0.28;
        } else if (cid === ac) {
          mat.opacity = 1.0;
          mat.size = 0.36;
        } else {
          mat.opacity = 0.06;
          mat.size = 0.20;
        }
        mat.needsUpdate = true;
      }

      renderer.render(scene, camera);
    };
    animate();

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      mounted = false;
      cancelAnimationFrame(animFrame);
      clearTimeout(resumeTimer);
      renderer.domElement.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.domElement.removeEventListener("mousemove", onCanvasMouseMove);
      renderer.domElement.removeEventListener("click", onCanvasClick);
      renderer.domElement.removeEventListener("mouseleave", onMouseLeave);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
      pointTex.dispose();
      for (const cloud of pointClouds) {
        cloud.geometry.dispose();
        cloud.material.dispose();
      }
    };
  }, [points]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        borderRadius: "var(--border-radius-md)",
        overflow: "hidden",
        cursor: isDraggingCursor ? "grabbing" : "grab",
        background: "#0d0c0b",
      }}
    >
      {tooltip && (
        <div style={{
          position: "absolute",
          left: Math.min(tooltip.x + 12, (containerRef.current?.clientWidth ?? 300) - 160),
          top: Math.max(tooltip.y - 14, 4),
          background: "rgba(13,12,11,0.93)",
          border: `1px solid ${CLUSTER_COLORS[tooltip.cluster]}55`,
          borderRadius: "var(--border-radius-md)",
          padding: "7px 11px",
          fontSize: 11,
          lineHeight: 1.55,
          pointerEvents: "none",
          zIndex: 20,
          maxWidth: 155,
          color: "#e0dfd9",
          backdropFilter: "blur(4px)",
        }}>
          <div style={{ fontWeight: 500, marginBottom: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {tooltip.title}
          </div>
          {tooltip.artist && (
            <div style={{ color: "#888780", fontSize: 10, marginBottom: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {tooltip.artist}
            </div>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 7, height: 7, borderRadius: 99, background: CLUSTER_COLORS[tooltip.cluster], display: "inline-block", flexShrink: 0 }} />
            <span style={{ color: CLUSTER_COLORS[tooltip.cluster], fontSize: 10 }}>
              {CLUSTER_LABELS[tooltip.cluster]}
            </span>
            {tooltip.year && (
              <span style={{ color: "#666460", fontSize: 10, marginLeft: 2 }}>{tooltip.year}</span>
            )}
          </div>
        </div>
      )}

      {/* Interaction hint overlay */}
      <div style={{
        position: "absolute",
        bottom: 8,
        right: 10,
        fontSize: 10,
        color: "#444340",
        pointerEvents: "none",
        userSelect: "none",
        letterSpacing: "0.04em",
      }}>
        drag · scroll
      </div>
    </div>
  );
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
  const features = generateFeatureData(activeCluster);
  const words = getWordCloud(activeCluster);

  const featureEntries = [
    { label: "Energy", key: "energy" },
    { label: "Danceability", key: "danceability" },
    { label: "Loudness", key: "loudness" },
    { label: "Acousticness", key: "acousticness" },
    { label: "Valence", key: "valence" },
    { label: "Tempo", key: "tempo" },
  ];

  const clusterColor = CLUSTER_COLORS[activeCluster] ?? CLUSTER_COLORS[0];

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
                Artist tags — {activeCluster !== null ? CLUSTER_LABELS[activeCluster] : "all clusters"}
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

          {/* 3D Cluster Card */}
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "16px 20px",
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}>
            {/* Card header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                UMAP 3D — song clusters
              </div>
              <div style={{
                fontSize: 10,
                color: "var(--color-text-secondary)",
                background: "var(--color-background-secondary)",
                border: "0.5px solid var(--color-border-tertiary)",
                borderRadius: 99,
                padding: "2px 8px",
                letterSpacing: "0.03em",
              }}>
                {activeCluster !== null ? (
                  <span style={{ color: clusterColor }}>● {CLUSTER_LABELS[activeCluster]}</span>
                ) : (
                  "all clusters"
                )}
              </div>
            </div>

            {/* 3D scatter */}
            <ClusterScatter3D
              points={clusterPoints}
              activeCluster={activeCluster}
              onClusterClick={i => setActiveCluster(activeCluster === i ? null : i)}
            />

            {/* Cluster swatches */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "5px 10px" }}>
              {CLUSTER_LABELS.map((label, i) => (
                <button
                  key={i}
                  onClick={() => setActiveCluster(activeCluster === i ? null : i)}
                  style={{
                    display: "flex", alignItems: "center", gap: 4,
                    fontSize: 11,
                    background: activeCluster === i ? CLUSTER_COLORS[i] + "18" : "none",
                    border: activeCluster === i ? `0.5px solid ${CLUSTER_COLORS[i]}55` : "0.5px solid transparent",
                    borderRadius: 99,
                    padding: "2px 7px 2px 5px",
                    cursor: "pointer",
                    color: activeCluster === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                    fontWeight: activeCluster === i ? 500 : 400,
                    opacity: activeCluster !== null && activeCluster !== i ? 0.45 : 1,
                    transition: "all 0.2s",
                  }}
                >
                  <span style={{ width: 6, height: 6, borderRadius: 99, background: CLUSTER_COLORS[i], display: "inline-block", flexShrink: 0 }} />
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
