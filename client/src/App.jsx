import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import * as THREE from "three";
import { fetchDashboardData } from "./api";

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
  "Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3",
  "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7",
];

const COLOR_ALL    = "#58A4B0";   // all-clusters overview color
const AXIS_COLORS  = ["#b03030", "#2a9a50", "#2a62a8"];
const AXIS_NAMES   = ["UMAP-1", "UMAP-2", "UMAP-3"];

const EMPTY_AUDIO_FEATURES = {
  features: {},
  display: {},
};

const EMPTY_DASHBOARD_DATA = {
  clusterData: [],
  themeRiverData: [],
  audioFeatureData: EMPTY_AUDIO_FEATURES,
  wordCloudData: {},
  stats: {
    totalSongs: 0,
    clusteredSongs: 0,
    clusters: 0,
  },
};

function getFeatureData(audioFeatureData, clusterId) {
  return audioFeatureData.features?.[String(clusterId)] ?? {};
}

function getWordCloud(wordCloudData, clusterId) {
  const key = clusterId === null ? "all" : String(clusterId);
  return wordCloudData[key] ?? [];
}

// ─── Soft-circle sprite texture ───────────────────────────────────────────────
function createPointTexture() {
  const sz = 64;
  const canvas = document.createElement("canvas");
  canvas.width = sz; canvas.height = sz;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(sz/2, sz/2, 0, sz/2, sz/2, sz/2);
  g.addColorStop(0.00, "rgba(255,255,255,1.0)");
  g.addColorStop(0.55, "rgba(255,255,255,0.95)");
  g.addColorStop(0.78, "rgba(255,255,255,0.4)");
  g.addColorStop(0.92, "rgba(255,255,255,0.05)");
  g.addColorStop(1.00, "rgba(255,255,255,0.0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, sz, sz);
  return new THREE.CanvasTexture(canvas);
}

// ─── 3-D Cluster Scatter ──────────────────────────────────────────────────────
function ClusterScatter3D({ points, activeCluster, onClusterClick }) {
  const containerRef    = useRef(null);
  const axisLabelRefs   = useRef([null, null, null]);
  const [tooltip, setTooltip]               = useState(null);
  const [isDraggingCursor, setIsDragCursor] = useState(false);
  const activeClusterRef = useRef(activeCluster);
  const onClickRef       = useRef(onClusterClick);

  useEffect(() => { activeClusterRef.current = activeCluster; }, [activeCluster]);
  useEffect(() => { onClickRef.current = onClusterClick; },    [onClusterClick]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !points.length) return;

    let mounted = true;
    const W = container.clientWidth  || 300;
    const H = container.clientHeight || 340;

    // ── Scene ─────────────────────────────────────────────────────────────────
    const BG = 0x080910;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.FogExp2(BG, 0.022);

    const camera = new THREE.PerspectiveCamera(52, W / H, 0.1, 200);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.display = "block";
    container.appendChild(renderer.domElement);

    // ── Normalise coords ──────────────────────────────────────────────────────
    const xs = points.map(p => p.x), ys = points.map(p => p.y), zs = points.map(p => p.z);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const zMin = Math.min(...zs), zMax = Math.max(...zs);
    const SPAN = 12, HALF = SPAN / 2;
    const norm = (v, mn, mx) => ((v - mn) / (mx - mn) - 0.5) * SPAN;

    const lookAt = new THREE.Vector3(0, 0, 0);

    // ── Grid box (3 planes) ───────────────────────────────────────────────────
    const applyGridMat = (g, opacity) => {
      const setM = m => { m.transparent = true; m.opacity = opacity; m.depthWrite = false; };
      Array.isArray(g.material) ? g.material.forEach(setM) : setM(g.material);
      return g;
    };

    const gridFloor = new THREE.GridHelper(SPAN, 10, 0x1c2840, 0x121a28);
    gridFloor.position.y = -HALF;
    scene.add(applyGridMat(gridFloor, 0.55));

    const gridBack = new THREE.GridHelper(SPAN, 10, 0x1c2840, 0x121a28);
    gridBack.rotation.x = Math.PI / 2;
    gridBack.position.z = -HALF;
    scene.add(applyGridMat(gridBack, 0.35));

    const gridLeft = new THREE.GridHelper(SPAN, 10, 0x1c2840, 0x121a28);
    gridLeft.rotation.z = Math.PI / 2;
    gridLeft.position.x = -HALF;
    scene.add(applyGridMat(gridLeft, 0.35));

    // ── Axis lines ────────────────────────────────────────────────────────────
    const origin   = new THREE.Vector3(-HALF, -HALF, -HALF);
    const AXLEN    = SPAN + 1.4;
    const axisEnds = [
      new THREE.Vector3(-HALF + AXLEN, -HALF,       -HALF      ),  // X
      new THREE.Vector3(-HALF,         -HALF + AXLEN, -HALF      ),  // Y
      new THREE.Vector3(-HALF,         -HALF,       -HALF + AXLEN),  // Z
    ];
    const axisHex = [0xaa2828, 0x28aa50, 0x2858aa];

    axisEnds.forEach((end, i) => {
      const geo = new THREE.BufferGeometry().setFromPoints([origin, end]);
      const mat = new THREE.LineBasicMaterial({ color: axisHex[i], transparent: true, opacity: 0.72 });
      scene.add(new THREE.Line(geo, mat));

      // Small tick arrowhead — a thin cone at tip
      const coneGeo = new THREE.ConeGeometry(0.12, 0.5, 6);
      const coneMat = new THREE.MeshBasicMaterial({ color: axisHex[i], transparent: true, opacity: 0.75 });
      const cone = new THREE.Mesh(coneGeo, coneMat);
      cone.position.copy(end);
      // Align cone along axis direction
      if (i === 0) cone.rotation.z = -Math.PI / 2;
      if (i === 2) cone.rotation.x =  Math.PI / 2;
      scene.add(cone);
    });

    // Label world positions (slightly beyond arrow tip)
    const LABEL_WORLD = axisEnds.map((e, i) => {
      const v = e.clone();
      if (i === 0) v.x += 0.5;
      if (i === 1) v.y += 0.5;
      if (i === 2) v.z += 0.5;
      return v;
    });
    const projVec = new THREE.Vector3();

    // ── Point clouds ──────────────────────────────────────────────────────────
    const pointTex = createPointTexture();
    const clustersMap = {};
    for (const p of points) {
      (clustersMap[p.cluster] ??= []).push(p);
    }

    const pointClouds = [];
    for (const cid of Object.keys(clustersMap).map(Number).sort()) {
      const pts = clustersMap[cid];
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        pos[i*3]   = norm(pts[i].x, xMin, xMax);
        pos[i*3+1] = norm(pts[i].y, yMin, yMax);
        pos[i*3+2] = norm(pts[i].z, zMin, zMax);
      }
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));

      const mat = new THREE.PointsMaterial({
        color: new THREE.Color(CLUSTER_COLORS[cid]),
        size: 0.32,
        map: pointTex,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.88,
        depthWrite: false,
        alphaTest: 0.08,
        blending: THREE.NormalBlending,
      });

      const cloud = new THREE.Points(geo, mat);
      cloud.userData = { clusterId: cid, pts };
      scene.add(cloud);
      pointClouds.push(cloud);
    }

    // ── Orbit state ───────────────────────────────────────────────────────────
    let theta = 0.6, phi = 1.15, radius = 18;
    let isDragging = false, lastX = 0, lastY = 0;
    let autoRotate = true, resumeTimer = null;

    const updateCamera = () => {
      camera.position.set(
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.cos(theta),
      );
      camera.lookAt(lookAt);
    };
    updateCamera();

    const pauseAutoRotate = () => {
      autoRotate = false;
      clearTimeout(resumeTimer);
      resumeTimer = setTimeout(() => { autoRotate = true; }, 3500);
    };

    // ── Event handlers ────────────────────────────────────────────────────────
    const onMouseDown = e => {
      isDragging = true; lastX = e.clientX; lastY = e.clientY;
      pauseAutoRotate();
      if (mounted) setIsDragCursor(true);
    };
    const onMouseMove = e => {
      if (!isDragging) return;
      theta -= (e.clientX - lastX) * 0.007;
      phi = Math.max(0.12, Math.min(Math.PI - 0.12, phi - (e.clientY - lastY) * 0.007));
      lastX = e.clientX; lastY = e.clientY;
      updateCamera();
      if (mounted) setTooltip(null);
    };
    const onMouseUp = () => {
      isDragging = false;
      if (mounted) setIsDragCursor(false);
    };
    const onWheel = e => {
      e.preventDefault();
      radius = Math.max(8, Math.min(40, radius + e.deltaY * 0.04));
      updateCamera();
      pauseAutoRotate();
    };

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.38;
    const mouse = new THREE.Vector2();
    let hoverTick = 0;

    const onCanvasMouseMove = e => {
      if (isDragging) return;
      if (++hoverTick % 2 !== 0) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(pointClouds);
      if (hits.length > 0) {
        const pt = hits[0].object.userData.pts[hits[0].index];
        if (mounted) setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          title:   pt.title  || "—",
          artist:  pt.artist || "",
          cluster: pt.cluster,
          year:    pt.year > 0 ? pt.year : null,
        });
      } else if (mounted) setTooltip(null);
    };
    const onCanvasClick = e => {
      if (isDragging) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(pointClouds);
      if (hits.length > 0) onClickRef.current?.(hits[0].object.userData.clusterId);
    };

    renderer.domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
    renderer.domElement.addEventListener("mousemove", onCanvasMouseMove);
    renderer.domElement.addEventListener("click", onCanvasClick);
    renderer.domElement.addEventListener("mouseleave", () => mounted && setTooltip(null));

    // ── Animation loop ────────────────────────────────────────────────────────
    let animFrame;
    const animate = () => {
      animFrame = requestAnimationFrame(animate);

      if (autoRotate) { theta += 0.0025; updateCamera(); }

      // Cluster opacity
      const ac = activeClusterRef.current;
      for (const cloud of pointClouds) {
        const m = cloud.material;
        const cid = cloud.userData.clusterId;
        if (ac === null) {
          m.opacity = 0.88; m.size = 0.32;
        } else if (cid === ac) {
          m.opacity = 1.00; m.size = 0.40;
        } else {
          m.opacity = 0.12; m.size = 0.22;
        }
        m.needsUpdate = true;
      }

      // Axis label DOM positioning
      for (let i = 0; i < 3; i++) {
        const el = axisLabelRefs.current[i];
        if (!el) continue;
        projVec.copy(LABEL_WORLD[i]).project(camera);
        if (projVec.z >= 1) { el.style.opacity = "0"; continue; }
        el.style.left    = `${((projVec.x + 1) / 2) * W}px`;
        el.style.top     = `${((1 - projVec.y) / 2) * H}px`;
        el.style.opacity = "1";
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
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
      renderer.dispose();
      pointTex.dispose();
      for (const c of pointClouds) { c.geometry.dispose(); c.material.dispose(); }
    };
  }, [points]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        borderRadius: "var(--border-radius-md)",
        overflow: "hidden",
        cursor: isDraggingCursor ? "grabbing" : "grab",
        background: "#080910",
      }}
    >
      {/* Axis labels — positioned by animation loop */}
      {AXIS_NAMES.map((name, i) => (
        <span
          key={name}
          ref={el => axisLabelRefs.current[i] = el}
          style={{
            position: "absolute",
            transform: "translate(-50%, -50%)",
            fontSize: 10,
            fontWeight: 600,
            color: AXIS_COLORS[i],
            pointerEvents: "none",
            letterSpacing: "0.07em",
            opacity: 0,
            userSelect: "none",
            textShadow: `0 0 10px ${AXIS_COLORS[i]}99`,
          }}
        >
          {name}
        </span>
      ))}

      {/* Hover tooltip */}
      {tooltip && (
        <div style={{
          position: "absolute",
          left: Math.min(tooltip.x + 14, (containerRef.current?.clientWidth ?? 300) - 165),
          top:  Math.max(tooltip.y - 14, 6),
          background: "rgba(8,9,16,0.93)",
          border: `1px solid ${CLUSTER_COLORS[tooltip.cluster]}55`,
          borderRadius: "var(--border-radius-md)",
          padding: "7px 11px",
          fontSize: 11,
          lineHeight: 1.55,
          pointerEvents: "none",
          zIndex: 20,
          maxWidth: 160,
          color: "#e0dfd9",
          backdropFilter: "blur(4px)",
        }}>
          <div style={{ fontWeight: 500, marginBottom: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {tooltip.title}
          </div>
          {tooltip.artist && (
            <div style={{ color: "#888780", fontSize: 10, marginBottom: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {tooltip.artist}
            </div>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 7, height: 7, borderRadius: 99, background: CLUSTER_COLORS[tooltip.cluster], display: "inline-block", flexShrink: 0 }} />
            <span style={{ color: CLUSTER_COLORS[tooltip.cluster], fontSize: 10 }}>
              {CLUSTER_LABELS[tooltip.cluster]}
            </span>
            {tooltip.year && <span style={{ color: "#555360", fontSize: 10, marginLeft: 2 }}>{tooltip.year}</span>}
          </div>
        </div>
      )}

      {/* Hint */}
      <div style={{ position: "absolute", bottom: 8, right: 10, fontSize: 10, color: "#2a2b35", pointerEvents: "none", userSelect: "none", letterSpacing: "0.04em" }}>
        drag · scroll
      </div>
    </div>
  );
}

// ─── Fullscreen Overlay ───────────────────────────────────────────────────────
function FullscreenOverlay({ points, activeCluster, onClusterClick, onClose }) {
  // Close on ESC
  useEffect(() => {
    const handler = e => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const clusterColor = CLUSTER_COLORS[activeCluster] ?? "#888780";

  return (
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        background: "rgba(4,4,8,0.92)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        animation: "fsIn 0.18s ease",
        padding: "20px 24px",
        gap: 16,
      }}
    >
      {/* Top bar */}
      <div style={{
        width: "100%", maxWidth: 1100,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        animation: "fsPanel 0.22s ease",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: "#c8c7c0", letterSpacing: "0.04em" }}>
            UMAP 3D — Song Clusters
          </span>
          {activeCluster !== null && (
            <span style={{
              fontSize: 11, color: clusterColor,
              background: clusterColor + "18",
              border: `0.5px solid ${clusterColor}55`,
              borderRadius: 99, padding: "2px 9px",
            }}>
              ● {CLUSTER_LABELS[activeCluster]}
            </span>
          )}
        </div>

        {/* Close button */}
        <button
          onClick={onClose}
          title="Close (Esc)"
          style={{
            background: "rgba(255,255,255,0.06)",
            border: "0.5px solid rgba(255,255,255,0.14)",
            borderRadius: 8,
            color: "#c8c7c0",
            width: 32, height: 32,
            display: "flex", alignItems: "center", justifyContent: "center",
            cursor: "pointer",
            fontSize: 14,
            transition: "background 0.15s, border-color 0.15s",
          }}
          onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.12)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.28)"; }}
          onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.14)"; }}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M1 1L11 11M11 1L1 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      {/* 3D canvas */}
      <div style={{
        width: "100%", maxWidth: 1100,
        flex: 1, minHeight: 0,
        borderRadius: 12,
        overflow: "hidden",
        border: "0.5px solid rgba(255,255,255,0.07)",
        animation: "fsPanel 0.25s ease",
      }}>
        <ClusterScatter3D
          points={points}
          activeCluster={activeCluster}
          onClusterClick={onClusterClick}
        />
      </div>

      {/* Cluster legend */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: "6px 12px",
        justifyContent: "center",
        animation: "fsPanel 0.28s ease",
      }}>
        {/* All Clusters pill */}
        <button
          onClick={() => onClusterClick(null)}
          style={{
            display: "flex", alignItems: "center", gap: 5,
            fontSize: 12,
            background: activeCluster === null ? COLOR_ALL + "22" : "rgba(255,255,255,0.04)",
            border: activeCluster === null ? `0.5px solid ${COLOR_ALL}66` : "0.5px solid rgba(255,255,255,0.10)",
            borderRadius: 99, padding: "4px 12px 4px 8px", cursor: "pointer",
            color: activeCluster === null ? "#e0dfd9" : "#888780",
            fontWeight: activeCluster === null ? 500 : 400,
            transition: "all 0.18s",
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: 99, background: COLOR_ALL, flexShrink: 0, display: "inline-block" }} />
          All Clusters
        </button>
        {CLUSTER_LABELS.map((label, i) => (
          <button
            key={i}
            onClick={() => onClusterClick(i)}
            style={{
              display: "flex", alignItems: "center", gap: 5,
              fontSize: 12,
              background: activeCluster === i ? CLUSTER_COLORS[i] + "22" : "rgba(255,255,255,0.04)",
              border: activeCluster === i ? `0.5px solid ${CLUSTER_COLORS[i]}66` : "0.5px solid rgba(255,255,255,0.10)",
              borderRadius: 99,
              padding: "4px 12px 4px 8px",
              cursor: "pointer",
              color: activeCluster === i ? "#e0dfd9" : "#888780",
              fontWeight: activeCluster === i ? 500 : 400,
              opacity: activeCluster !== null && activeCluster !== i ? 0.45 : 1,
              transition: "all 0.18s",
            }}
          >
            <span style={{ width: 7, height: 7, borderRadius: 99, background: CLUSTER_COLORS[i], flexShrink: 0, display: "inline-block" }} />
            {label}
          </button>
        ))}
      </div>

      {/* ESC hint */}
      <div style={{ fontSize: 10, color: "#333440", letterSpacing: "0.05em", userSelect: "none" }}>
        Press Esc or click outside to close
      </div>
    </div>
  );
}

// ─── ThemeRiver ───────────────────────────────────────────────────────────────
function ThemeRiver({ data, activeCluster, onYearHover, onClusterClick }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    const el = svgRef.current;
    const W = el.clientWidth || 620;
    const H = 230;
    const L = 52;   // left margin — room for the Y-axis label
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
      .range([L, W - 20]);
    const yExtent = [
      d3.min(series, s => d3.min(s, d => d[0])),
      d3.max(series, s => d3.max(s, d => d[1])),
    ];
    const yScale = d3.scaleLinear().domain(yExtent).range([H - 22, 16]);

    // ── Wiggle centre baseline (aesthetic anchor) ─────────────────────────────
    const yCentre = yScale(0);
    svg.append("line")
      .attr("x1", L).attr("x2", W - 20)
      .attr("y1", yCentre).attr("y2", yCentre)
      .attr("stroke", "var(--color-border-tertiary)")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3 4")
      .attr("opacity", 0.6);

    // ── Bands ─────────────────────────────────────────────────────────────────
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
      .attr("opacity", (_, i) => activeCluster === null ? 0.82 : i === activeCluster ? 1 : 0.18)
      .style("cursor", "pointer")
      .on("mousemove", function (event, d) {
        const [mx] = d3.pointer(event, el);
        const year = Math.round(xScale.invert(mx));
        const row  = data.find(r => r.year === year);
        if (row) {
          const ci       = parseInt(d.key.split("_")[1]);
          const yearTotal = keys.reduce((s, k) => s + (row[k] || 0), 0);
          const pct      = yearTotal > 0 ? Math.round((row[d.key] / yearTotal) * 100) : 0;
          setTooltip({ x: mx, y: event.offsetY, year, cluster: ci, value: row[d.key], pct });
          onYearHover?.(year);
        }
      })
      .on("mouseleave", () => { setTooltip(null); onYearHover?.(null); })
      .on("click", function (event, d) {
        const ci = parseInt(d.key.split("_")[1]);
        onClusterClick?.(ci);
      });

    // ── X axis ────────────────────────────────────────────────────────────────
    const xAxis = d3.axisBottom(xScale).ticks(10).tickFormat(d3.format("d")).tickSize(0);
    svg.append("g")
      .attr("transform", `translate(0,${H - 20})`)
      .call(xAxis)
      .call(g => g.select(".domain").remove())
      .selectAll("text")
      .style("font-size", "11px")
      .style("fill", "var(--color-text-secondary)");

    // ── Y-axis label (rotated) ────────────────────────────────────────────────
    svg.append("text")
      .attr("transform", `rotate(-90)`)
      .attr("x", -(H / 2))
      .attr("y", 13)
      .attr("text-anchor", "middle")
      .style("font-size", "9.5px")
      .style("fill", "var(--color-text-secondary)")
      .style("letter-spacing", "0.05em")
      .text("← song count →");

  }, [data, activeCluster]);

  return (
    <div style={{ position: "relative" }}>
      <svg ref={svgRef} style={{ display: "block", width: "100%" }} />
      {tooltip && (
        <div style={{
          position: "absolute", left: tooltip.x + 12, top: tooltip.y - 10,
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderRadius: "var(--border-radius-md)",
          padding: "8px 12px", fontSize: 12, pointerEvents: "none", zIndex: 10,
        }}>
          <div style={{ fontWeight: 500, marginBottom: 5 }}>{tooltip.year}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: CLUSTER_COLORS[tooltip.cluster], display: "inline-block", flexShrink: 0 }} />
            <span style={{ color: "var(--color-text-secondary)" }}>{CLUSTER_LABELS[tooltip.cluster]}</span>
          </div>
          <div style={{ display: "flex", gap: 10, paddingLeft: 14 }}>
            <span style={{ fontWeight: 500 }}>{tooltip.value} songs</span>
            <span style={{ color: "var(--color-text-secondary)" }}>·</span>
            <span style={{ color: "var(--color-text-secondary)" }}>{tooltip.pct}% of year</span>
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
        <div style={{ height: "100%", width: `${value * 100}%`, background: color, borderRadius: 4, transition: "width 0.5s ease" }} />
      </div>
      <div style={{ width: 32, fontSize: 12, fontWeight: 500, flexShrink: 0 }}>{value.toFixed(2)}</div>
    </div>
  );
}

// ─── Word Cloud ───────────────────────────────────────────────────────────────
function WordCloud({ words, color }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 8px", alignItems: "baseline", minHeight: 80 }}>
      {[...words].sort((a, b) => b.w - a.w).map(({ word, w }) => (
        <span key={word} style={{
          fontSize: `${Math.round(11 + w * 18)}px`,
          fontWeight: w > 0.7 ? 500 : 400,
          color: w > 0.65 ? color : "var(--color-text-secondary)",
          lineHeight: 1.3,
          transition: "all 0.4s ease",
        }}>
          {word}
        </span>
      ))}
    </div>
  );
}

// ─── Expand / Collapse icon buttons ──────────────────────────────────────────
function IconBtn({ onClick, title, children, style }) {
  const [hover, setHover] = useState(false);
  return (
    <button
      onClick={onClick}
      title={title}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        background: hover ? "var(--color-background-secondary)" : "none",
        border: "0.5px solid " + (hover ? "var(--color-border-secondary)" : "transparent"),
        borderRadius: 6,
        color: "var(--color-text-secondary)",
        width: 26, height: 26,
        display: "flex", alignItems: "center", justifyContent: "center",
        cursor: "pointer",
        transition: "background 0.15s, border-color 0.15s",
        flexShrink: 0,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeCluster, setActiveCluster] = useState(0);
  const [hoveredYear,   setHoveredYear]   = useState(null);
  const [isFullscreen,  setIsFullscreen]  = useState(false);
  const [dashboardData, setDashboardData] = useState(EMPTY_DASHBOARD_DATA);
  const [loadState, setLoadState] = useState({ isLoading: true, error: null });

  useEffect(() => {
    let ignore = false;

    async function load() {
      try {
        const payload = await fetchDashboardData();
        if (ignore) return;
        setDashboardData({
          clusterData: payload.clusterData ?? [],
          themeRiverData: payload.themeRiverData ?? [],
          audioFeatureData: payload.audioFeatureData ?? EMPTY_AUDIO_FEATURES,
          wordCloudData: payload.wordCloudData ?? {},
          stats: payload.stats ?? EMPTY_DASHBOARD_DATA.stats,
        });
        setLoadState({ isLoading: false, error: null });
      } catch (error) {
        if (ignore) return;
        setLoadState({
          isLoading: false,
          error: error.message || "Failed to load dashboard data.",
        });
      }
    }

    load();
    return () => { ignore = true; };
  }, []);

  const { clusterData, themeRiverData, audioFeatureData, wordCloudData, stats } = dashboardData;
  const featureEntries = Object.entries(audioFeatureData.display ?? {}).map(
    ([key, label]) => ({ key, label }),
  );
  const features = getFeatureData(audioFeatureData, activeCluster ?? 0);
  const words = getWordCloud(wordCloudData, activeCluster);

  const clusterColor = activeCluster !== null ? CLUSTER_COLORS[activeCluster] : COLOR_ALL;

  const handleClusterClick = i =>
    setActiveCluster(i === null ? null : prev => prev === i ? null : i);

  if (loadState.isLoading) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        background: "var(--color-background-tertiary)",
        fontFamily: "var(--font-sans)",
        color: "var(--color-text-secondary)",
      }}>
        Loading dashboard data...
      </div>
    );
  }

  if (loadState.error) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        background: "var(--color-background-tertiary)",
        fontFamily: "var(--font-sans)",
        padding: 24,
      }}>
        <div style={{
          maxWidth: 560,
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
          padding: 24,
          color: "var(--color-text-primary)",
        }}>
          <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 8 }}>
            Failed to load dashboard data
          </div>
          <div style={{ color: "var(--color-text-secondary)", fontSize: 13, lineHeight: 1.5 }}>
            {loadState.error}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ fontFamily: "var(--font-sans)", background: "var(--color-background-tertiary)", minHeight: "100vh", padding: "0 0 2rem" }}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{
        background: "var(--color-background-primary)",
        borderBottom: "0.5px solid var(--color-border-tertiary)",
        padding: "14px 24px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <h1 style={{ fontSize: 16, fontWeight: 500, margin: 0 }}>Multi-modal Music Trend Analysis</h1>
          <span style={{
            fontSize: 12, color: "var(--color-text-secondary)",
            background: "var(--color-background-secondary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-md)", padding: "2px 10px",
          }}>1960 – 2010</span>
        </div>
        <div style={{
          fontSize: 13, padding: "5px 14px", borderRadius: 99,
          border: `0.5px solid ${clusterColor}`,
          background: clusterColor + "18", color: clusterColor, fontWeight: 500,
        }}>
          Audio + Lyrics + Artist Tags
        </div>
      </div>

      <div style={{ padding: "20px 24px", display: "grid", gridTemplateColumns: "1fr 320px", gap: 16 }}>

        {/* ── LEFT COLUMN ──────────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* ThemeRiver */}
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)", padding: "16px 20px",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                ThemeRiver — genre share by year · band width = song count
              </div>
              {hoveredYear && <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>year: {hoveredYear}</span>}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", marginBottom: 12 }}>
              {CLUSTER_LABELS.map((label, i) => (
                <button key={i} onClick={() => handleClusterClick(i)} style={{
                  display: "flex", alignItems: "center", gap: 5, fontSize: 12,
                  background: "none", border: "none", cursor: "pointer", padding: 0,
                  color: activeCluster === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  fontWeight: activeCluster === i ? 500 : 400,
                  opacity: activeCluster !== null && activeCluster !== i ? 0.5 : 1,
                  transition: "all 0.2s",
                }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: CLUSTER_COLORS[i], flexShrink: 0 }} />
                  {label}
                </button>
              ))}
            </div>
            <ThemeRiver data={themeRiverData} activeCluster={activeCluster} onYearHover={setHoveredYear} onClusterClick={handleClusterClick} />
          </div>

          {/* Audio Features + Word Cloud */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "16px 20px" }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 14 }}>
                Audio features — {activeCluster !== null ? CLUSTER_LABELS[activeCluster] : "all clusters"}
              </div>
              {featureEntries.map(({ label, key }) => (
                <AudioFeatureBar key={key} label={label} value={features[key] ?? 0} color={clusterColor} />
              ))}
            </div>
            <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "16px 20px" }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 14 }}>
                Artist tags — {activeCluster !== null ? CLUSTER_LABELS[activeCluster] : "all clusters"}
              </div>
              <WordCloud words={words} color={clusterColor} />
            </div>
          </div>
        </div>

        {/* ── RIGHT COLUMN ─────────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Stats */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { label: "Songs", value: stats.totalSongs.toLocaleString(), sub: `${stats.clusteredSongs.toLocaleString()} clustered` },
              { label: "Clusters", value: String(stats.clusters), sub: "audio + lyrics" },
            ].map(({ label, value, sub }) => (
              <div key={label} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px 14px" }}>
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
            display: "flex", flexDirection: "column", gap: 12,
          }}>
            {/* Card header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                UMAP 3D — song clusters
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                {activeCluster !== null && (
                  <span style={{ fontSize: 10, color: clusterColor }}>
                    ● {CLUSTER_LABELS[activeCluster]}
                  </span>
                )}
                {/* Expand to fullscreen */}
                <IconBtn onClick={() => setIsFullscreen(true)} title="Expand to fullscreen">
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M1 4.5V1H4.5M7.5 1H11V4.5M11 7.5V11H7.5M4.5 11H1V7.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </IconBtn>
              </div>
            </div>

            {/* Compact 3D view */}
            <div style={{ height: 300 }}>
              <ClusterScatter3D
                points={clusterData}
                activeCluster={activeCluster}
                onClusterClick={handleClusterClick}
              />
            </div>

            {/* Cluster swatches */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "5px 8px" }}>
              {/* All Clusters pill */}
              <button onClick={() => setActiveCluster(null)} style={{
                display: "flex", alignItems: "center", gap: 4, fontSize: 11,
                background: activeCluster === null ? COLOR_ALL + "18" : "none",
                border: activeCluster === null ? `0.5px solid ${COLOR_ALL}55` : "0.5px solid transparent",
                borderRadius: 99, padding: "2px 7px 2px 5px", cursor: "pointer",
                color: activeCluster === null ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                fontWeight: activeCluster === null ? 500 : 400,
                transition: "all 0.2s",
              }}>
                <span style={{ width: 6, height: 6, borderRadius: 99, background: COLOR_ALL, display: "inline-block", flexShrink: 0 }} />
                All Clusters
              </button>
              {CLUSTER_LABELS.map((label, i) => (
                <button key={i} onClick={() => handleClusterClick(i)} style={{
                  display: "flex", alignItems: "center", gap: 4, fontSize: 11,
                  background: activeCluster === i ? CLUSTER_COLORS[i] + "18" : "none",
                  border: activeCluster === i ? `0.5px solid ${CLUSTER_COLORS[i]}55` : "0.5px solid transparent",
                  borderRadius: 99, padding: "2px 7px 2px 5px", cursor: "pointer",
                  color: activeCluster === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  fontWeight: activeCluster === i ? 500 : 400,
                  opacity: activeCluster !== null && activeCluster !== i ? 0.45 : 1,
                  transition: "all 0.2s",
                }}>
                  <span style={{ width: 6, height: 6, borderRadius: 99, background: CLUSTER_COLORS[i], display: "inline-block", flexShrink: 0 }} />
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Fullscreen overlay ────────────────────────────────────────────── */}
      {isFullscreen && (
        <FullscreenOverlay
          points={clusterData}
          activeCluster={activeCluster}
          onClusterClick={handleClusterClick}
          onClose={() => setIsFullscreen(false)}
        />
      )}
    </div>
  );
}

