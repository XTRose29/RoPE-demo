const state = {
  positionEncoding: false,
  rotatePosition: 3,
  queryPosition: 2,
  keyPosition: 5,
  multiPosition: 8,
  attentionMode: "rope",
  selectedQuery: 4
};

const baseQuery2D = [1.0, 0.4];
const baseKey2D = [0.85, -0.1];
const theta2D = 0.45;
const sentenceA = ["The", "cat", "chased", "the", "dog"];
const sentenceB = ["The", "dog", "chased", "the", "cat"];
const attentionTokens = ["The", "small", "cat", "quietly", "chased", "the", "dog", "near", "the", "river"];
const contentScores = [0.14, 0.2, 0.34, 0.62, 1.0, 0.58, 0.31, 0.16, 0.09, 0.05];

function rotate2D(x, y, angle) {
  return [
    x * Math.cos(angle) - y * Math.sin(angle),
    x * Math.sin(angle) + y * Math.cos(angle)
  ];
}

function theta(i, d) {
  return Math.pow(10000, -2 * i / d);
}

function applyRoPE(vec, pos) {
  const d = vec.length;
  const out = [...vec];

  for (let i = 0; i < d; i += 2) {
    const angle = pos * theta(i / 2, d);
    const [x, y] = rotate2D(vec[i], vec[i + 1], angle);
    out[i] = x;
    out[i + 1] = y;
  }

  return out;
}

function dot(a, b) {
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

function fmt(value) {
  return Number(value).toFixed(2);
}

function normalize(value, min, max) {
  if (max === min) return 0;
  return (value - min) / (max - min);
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function drawGrid(svg, cx, cy, radius) {
  svg.innerHTML = "";
  for (let i = -2; i <= 2; i++) {
    const offset = i * (radius / 2);
    svg.appendChild(svgEl("line", { class: "grid-line", x1: cx - radius, y1: cy + offset, x2: cx + radius, y2: cy + offset }));
    svg.appendChild(svgEl("line", { class: "grid-line", x1: cx + offset, y1: cy - radius, x2: cx + offset, y2: cy + radius }));
  }
  svg.appendChild(svgEl("line", { class: "axis", x1: cx - radius, y1: cy, x2: cx + radius, y2: cy }));
  svg.appendChild(svgEl("line", { class: "axis", x1: cx, y1: cy + radius, x2: cx, y2: cy - radius }));
  svg.appendChild(svgEl("circle", { cx, cy, r: radius, fill: "none", stroke: "#e2e8f0", "stroke-width": 2 }));
}

function drawVector(svg, cx, cy, vec, scale, color, marker, label, offsetY = -10) {
  const x2 = cx + vec[0] * scale;
  const y2 = cy - vec[1] * scale;
  svg.appendChild(svgEl("line", {
    x1: cx,
    y1: cy,
    x2,
    y2,
    stroke: color,
    "stroke-width": 5,
    "stroke-linecap": "round",
    "marker-end": `url(#${marker})`
  }));
  const text = svgEl("text", {
    x: x2 + 10,
    y: y2 + offsetY,
    fill: color,
    class: "vector-label"
  });
  text.textContent = label;
  svg.appendChild(text);
}

function ensureMarkers(svg) {
  const defs = svgEl("defs");
  [
    ["arrowBlueLocal", "#2563eb"],
    ["arrowGreenLocal", "#16a34a"],
    ["arrowOrangeLocal", "#f59e0b"],
    ["arrowPurpleLocal", "#7c3aed"],
    ["arrowGrayLocal", "#94a3b8"]
  ].forEach(([id, color]) => {
    const marker = svgEl("marker", { id, markerWidth: 10, markerHeight: 10, refX: 8, refY: 3, orient: "auto" });
    marker.appendChild(svgEl("path", { d: "M0,0 L9,3 L0,6 Z", fill: color }));
    defs.appendChild(marker);
  });
  svg.appendChild(defs);
}

function renderTokens() {
  const a = document.querySelector("#sentenceA");
  const b = document.querySelector("#sentenceB");
  a.innerHTML = "";
  b.innerHTML = "";
  [sentenceA, sentenceB].forEach((sentence, sentenceIndex) => {
    const target = sentenceIndex === 0 ? a : b;
    sentence.forEach((word, index) => {
      const card = document.createElement("div");
      card.className = "word-card";
      if (state.positionEncoding) card.classList.add("with-position");
      if (sentenceIndex === 1 && (index === 1 || index === 4)) card.classList.add("reordered");
      card.dataset.pos = index;
      card.textContent = word;
      target.appendChild(card);
    });
  });

  document.querySelector("#whyNote").textContent = state.positionEncoding
    ? "Position tags let the same words carry different order information."
    : "Self-attention sees token content, but position must be injected separately.";

  const toggle = document.querySelector("#positionToggle");
  toggle.setAttribute("aria-pressed", String(state.positionEncoding));
  toggle.querySelectorAll("span").forEach((span) => {
    span.classList.toggle("selected", span.dataset.mode === (state.positionEncoding ? "on" : "off"));
  });
}

function renderRotate() {
  const svg = document.querySelector("#rotatePlot");
  const cx = 320;
  const cy = 270;
  const radius = 190;
  drawGrid(svg, cx, cy, radius);
  ensureMarkers(svg);

  const angle = state.rotatePosition * theta2D;
  const rotated = rotate2D(baseQuery2D[0], baseQuery2D[1], angle);
  drawVector(svg, cx, cy, baseQuery2D, 125, "#94a3b8", "arrowGrayLocal", "q before", 18);
  drawVector(svg, cx, cy, rotated, 125, "#2563eb", "arrowBlueLocal", "q_p");

  const arc = svgEl("path", {
    d: `M ${cx + 58} ${cy} A 58 58 0 ${Math.abs(angle) > Math.PI ? 1 : 0} 0 ${cx + Math.cos(-angle) * 58} ${cy + Math.sin(-angle) * 58}`,
    fill: "none",
    stroke: "#7c3aed",
    "stroke-width": 4,
    "stroke-linecap": "round"
  });
  svg.appendChild(arc);

  const angleText = svgEl("text", { x: 36, y: 52, fill: "#7c3aed", class: "vector-label" });
  angleText.textContent = `angle = ${fmt(angle)} rad`;
  svg.appendChild(angleText);

  document.querySelector("#positionValue").textContent = state.rotatePosition;
  document.querySelector("#rotateStats").innerHTML = `
    <div>Before rotation: <strong>q = (${fmt(baseQuery2D[0])}, ${fmt(baseQuery2D[1])})</strong></div>
    <div>Position p = <strong>${state.rotatePosition}</strong></div>
    <div>Angle = pθ = <strong>${state.rotatePosition}θ = ${fmt(angle)} rad</strong></div>
    <div>After rotation: <strong>q<sub>p</sub> = (${fmt(rotated[0])}, ${fmt(rotated[1])})</strong></div>
  `;
}

function renderRelative() {
  const svg = document.querySelector("#relativePlot");
  const cx = 320;
  const cy = 230;
  const radius = 165;
  drawGrid(svg, cx, cy, radius);
  ensureMarkers(svg);

  const qm = rotate2D(baseQuery2D[0], baseQuery2D[1], state.queryPosition * theta2D);
  const kn = rotate2D(baseKey2D[0], baseKey2D[1], state.keyPosition * theta2D);
  const score = dot(qm, kn);
  const delta = state.keyPosition - state.queryPosition;

  drawVector(svg, cx, cy, qm, 135, "#2563eb", "arrowBlueLocal", "q_m", -12);
  drawVector(svg, cx, cy, kn, 135, "#16a34a", "arrowGreenLocal", "k_n", 22);

  const deltaText = svgEl("text", { x: 28, y: 42, fill: "#f59e0b", class: "vector-label" });
  deltaText.textContent = `same pattern when Δ = ${delta}`;
  svg.appendChild(deltaText);

  document.querySelector("#queryValue").textContent = state.queryPosition;
  document.querySelector("#keyValue").textContent = state.keyPosition;
  document.querySelector("#deltaValue").textContent = delta;
  document.querySelector("#scoreValue").textContent = fmt(score);
}

function renderClocks() {
  const root = document.querySelector("#clocks");
  root.innerHTML = "";
  const d = 8;
  for (let pair = 0; pair < 4; pair++) {
    const wrapper = document.createElement("div");
    wrapper.className = "clock";
    const label = document.createElement("p");
    label.className = "clock-title";
    const speed = pair === 0 ? "fast" : pair === 1 ? "slower" : pair === 2 ? "slow" : "slowest";
    label.textContent = `Pair ${pair + 1}: θ${pair + 1} ${speed}`;

    const svg = svgEl("svg", { viewBox: "0 0 180 180", role: "img", "aria-label": `RoPE pair ${pair + 1}` });
    const cx = 90;
    const cy = 88;
    const r = 62;
    svg.appendChild(svgEl("circle", { cx, cy, r, fill: "white", stroke: "#dbe2ea", "stroke-width": 3 }));
    svg.appendChild(svgEl("line", { x1: cx - r, y1: cy, x2: cx + r, y2: cy, stroke: "#edf2f7", "stroke-width": 2 }));
    svg.appendChild(svgEl("line", { x1: cx, y1: cy - r, x2: cx, y2: cy + r, stroke: "#edf2f7", "stroke-width": 2 }));
    ensureMarkers(svg);

    const angle = state.multiPosition * theta(pair, d);
    const hand = [Math.cos(angle), Math.sin(angle)];
    const colors = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed"];
    const markers = ["arrowBlueLocal", "arrowGreenLocal", "arrowOrangeLocal", "arrowPurpleLocal"];
    drawVector(svg, cx, cy, hand, 52, colors[pair], markers[pair], "");

    const value = svgEl("text", { x: cx, y: 168, fill: "#64748b", "text-anchor": "middle", "font-size": 12, "font-weight": 800 });
    value.textContent = `θ=${theta(pair, d).toPrecision(2)}`;
    svg.appendChild(value);

    wrapper.appendChild(svg);
    wrapper.appendChild(label);
    root.appendChild(wrapper);
  }
  document.querySelector("#multiPositionValue").textContent = state.multiPosition;
}

function ropeSimilarity(distance) {
  const d = 8;
  let total = 0;
  for (let pair = 0; pair < 4; pair++) {
    total += Math.cos(distance * theta(pair, d));
  }
  return total / 4;
}

function modeScore(index) {
  const distance = Math.abs(index - state.selectedQuery);
  if (state.attentionMode === "none") {
    return contentScores[index];
  }
  if (state.attentionMode === "sin") {
    return 0.62 * contentScores[index] + 0.38 * (0.5 + 0.5 * Math.cos(distance * 0.85));
  }
  const sim = Math.max(0, ropeSimilarity(distance));
  return 0.55 * contentScores[index] + 0.45 * Math.pow(sim, 1.5);
}

function renderHeatmap() {
  const select = document.querySelector("#queryTokenSelect");
  if (!select.children.length) {
    attentionTokens.forEach((token, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${index}: ${token}`;
      select.appendChild(option);
    });
  }
  select.value = String(state.selectedQuery);

  const rawScores = attentionTokens.map((_, index) => modeScore(index));
  const max = Math.max(...rawScores);
  const min = Math.min(...rawScores);
  const scores = rawScores.map((score) => normalize(score, min, max));
  const root = document.querySelector("#heatmap");
  root.innerHTML = "";

  attentionTokens.forEach((token, index) => {
    const cell = document.createElement("div");
    const normalized = scores[index];
    const alpha = 0.12 + normalized * 0.68;
    cell.className = "heat-cell";
    if (index === state.selectedQuery) cell.classList.add("query");
    cell.style.background = `rgba(37, 99, 235, ${alpha})`;
    cell.innerHTML = `<span class="heat-token">${token}</span><span class="heat-score">${fmt(normalized)}</span>`;
    root.appendChild(cell);
  });

  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.attentionMode);
  });
}

function renderDecay() {
  const svg = document.querySelector("#decayPlot");
  svg.innerHTML = "";
  const w = 640;
  const h = 160;
  const left = 48;
  const right = 22;
  const top = 18;
  const bottom = 34;
  const innerW = w - left - right;
  const innerH = h - top - bottom;

  svg.appendChild(svgEl("line", { x1: left, y1: h - bottom, x2: w - right, y2: h - bottom, stroke: "#cbd5e1", "stroke-width": 2 }));
  svg.appendChild(svgEl("line", { x1: left, y1: top, x2: left, y2: h - bottom, stroke: "#cbd5e1", "stroke-width": 2 }));

  const points = [];
  for (let dist = 0; dist <= 24; dist++) {
    const x = left + (dist / 24) * innerW;
    const sim = Math.max(0, ropeSimilarity(dist));
    const y = top + (1 - sim) * innerH;
    points.push([x, y]);
  }

  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");
  svg.appendChild(svgEl("path", { d: path, fill: "none", stroke: "#f59e0b", "stroke-width": 4, "stroke-linecap": "round" }));

  [["0", left, h - 12], ["|m-n|", w - 70, h - 12], ["similarity", 10, 22]].forEach(([text, x, y]) => {
    const label = svgEl("text", { x, y, fill: "#64748b", "font-size": 12, "font-weight": 800 });
    label.textContent = text;
    svg.appendChild(label);
  });
}

function renderAttention() {
  renderClocks();
  renderHeatmap();
  renderDecay();
}

function renderAll() {
  renderTokens();
  renderRotate();
  renderRelative();
  renderAttention();
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((button) => button.classList.remove("active"));
      document.querySelectorAll(".stage").forEach((stage) => stage.classList.remove("active"));
      tab.classList.add("active");
      document.querySelector(`#${tab.dataset.tab}`).classList.add("active");
    });
  });

  document.querySelector("#positionToggle").addEventListener("click", (event) => {
    const mode = event.target.dataset.mode;
    state.positionEncoding = mode ? mode === "on" : !state.positionEncoding;
    renderTokens();
  });

  document.querySelector("#positionSlider").addEventListener("input", (event) => {
    state.rotatePosition = Number(event.target.value);
    renderRotate();
  });

  document.querySelector("#querySlider").addEventListener("input", (event) => {
    state.queryPosition = Number(event.target.value);
    renderRelative();
  });

  document.querySelector("#keySlider").addEventListener("input", (event) => {
    state.keyPosition = Number(event.target.value);
    renderRelative();
  });

  document.querySelector("#shiftBoth").addEventListener("click", () => {
    if (Math.max(state.queryPosition, state.keyPosition) < 10) {
      state.queryPosition += 1;
      state.keyPosition += 1;
    }
    document.querySelector("#querySlider").value = state.queryPosition;
    document.querySelector("#keySlider").value = state.keyPosition;
    renderRelative();
  });

  document.querySelector("#multiPosition").addEventListener("input", (event) => {
    state.multiPosition = Number(event.target.value);
    renderAttention();
  });

  document.querySelector("#queryTokenSelect").addEventListener("change", (event) => {
    state.selectedQuery = Number(event.target.value);
    renderHeatmap();
  });

  document.querySelectorAll(".mode").forEach((button) => {
    button.addEventListener("click", () => {
      state.attentionMode = button.dataset.mode;
      renderHeatmap();
    });
  });
}

bindEvents();
renderAll();
