const state = {
  positionEncoding: false,
  rotatePosition: 3,
  queryPosition: 2,
  keyPosition: 5,
  multiPosition: 8,
  attentionMode: "rope",
  attentionExample: "sentence",
  selectedQuery: 4,
  trainingStep: 0,
  trainingMetric: "accuracy",
  selectedTrainingMethod: "all"
};

const baseQuery2D = [1.0, 0.4];
const baseKey2D = [0.85, -0.1];
const theta2D = 0.45;
const maxMultiPosition = 48;
const frequencyClockPairs = 4;
const frequencyClockDimension = frequencyClockPairs * 2;
const similarityPairs = 8;
const similarityDimension = similarityPairs * 2;
const generatedTrainingReplay = window.TINY_TRAINING_REPLAY || null;
const trainingTotalSteps = generatedTrainingReplay?.metadata?.totalSteps || 100;
const sentenceA = ["The", "cat", "chased", "the", "dog"];
const sentenceB = ["The", "dog", "chased", "the", "cat"];
const attentionTokens = ["The", "small", "cat", "quietly", "chased", "the", "dog", "near", "the", "river"];
const contentScores = [0.14, 0.2, 0.34, 0.62, 1.0, 0.58, 0.31, 0.16, 0.09, 0.05];
const longAttentionTokens = Array.from({ length: 48 }, (_, index) => `pos ${index}`);
const longContentScores = longAttentionTokens.map((_, index) => 0.5 + 0.22 * Math.sin(index * 0.73) + 0.12 * Math.cos(index * 0.31));
let trainingTimer = null;
const trainingMethods = [
  {
    id: "none",
    name: "No position",
    color: "#94a3b8",
    short: "Cannot reliably tell order",
    note: "Token content is visible, but order is not explicitly encoded. A and B can be detected, but their order is hard to distinguish."
  },
  {
    id: "sin",
    name: "Additive sinusoidal",
    color: "#2563eb",
    short: "Generalizes better than learned absolute",
    note: "Adds a fixed sinusoidal position vector to each token representation."
  },
  {
    id: "learned",
    name: "Learned absolute",
    color: "#f59e0b",
    short: "Learns fixed positions",
    note: "Learns a separate embedding for each absolute position. Works well on seen positions, but may struggle on unseen longer positions."
  },
  {
    id: "rope",
    name: "RoPE",
    color: "#7c3aed",
    short: "Relative/rotation-based position helps longer lengths",
    note: "Rotates query/key vectors by position. The attention score naturally depends on relative distance."
  }
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smooth01(value) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function makeTrainingReplay() {
  const config = {
    none: { train: 0.56, test: 0.52, loss: 0.68, speed: 1.2 },
    sin: { train: 0.94, test: 0.82, loss: 0.31, speed: 3.0 },
    learned: { train: 0.96, test: 0.66, loss: 0.42, speed: 3.5 },
    rope: { train: 0.95, test: 0.90, loss: 0.25, speed: 3.2 }
  };

  return Object.fromEntries(trainingMethods.map((method) => {
    const cfg = config[method.id];
    const points = Array.from({ length: trainingTotalSteps + 1 }, (_, step) => {
      const t = step / trainingTotalSteps;
      const learn = smooth01(1 - Math.exp(-cfg.speed * t));
      const wiggle = 0.008 * Math.sin(step * 0.23 + method.id.length);
      const trainAcc = clamp(0.5 + (cfg.train - 0.5) * learn + wiggle, 0.48, 0.98);
      const testAcc = clamp(0.5 + (cfg.test - 0.5) * learn + 0.7 * wiggle, 0.47, 0.94);
      const loss = clamp(0.69 - (0.69 - cfg.loss) * learn + 0.01 * Math.cos(step * 0.19 + method.id.length), 0.2, 0.72);
      return { step, trainAcc, testAcc, loss };
    });
    return [method.id, points];
  }));
}

const trainingReplay = generatedTrainingReplay
  ? Object.fromEntries(trainingMethods.map((method) => [method.id, generatedTrainingReplay.methods[method.id].curve]))
  : makeTrainingReplay();

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

function fmtTheta(value) {
  if (value >= 0.01) return Number(value).toPrecision(2);
  return Number(value).toPrecision(2);
}

function fmtAngle(value) {
  if (Math.abs(value) >= 1) return Number(value).toFixed(2);
  return Number(value).toPrecision(2);
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
    ["arrowTealLocal", "#0f766e"],
    ["arrowGreenLocal", "#16a34a"],
    ["arrowLimeLocal", "#84cc16"],
    ["arrowOrangeLocal", "#f59e0b"],
    ["arrowBurntLocal", "#ea580c"],
    ["arrowPurpleLocal", "#7c3aed"],
    ["arrowRoseLocal", "#db2777"],
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
  const d = frequencyClockDimension;
  for (let pair = 0; pair < frequencyClockPairs; pair++) {
    const wrapper = document.createElement("div");
    wrapper.className = "clock";
    const label = document.createElement("p");
    label.className = "clock-title";
    const speed = pair < 2 ? "fast rotation" : pair < 4 ? "medium/fast rotation" : pair < 6 ? "slow rotation" : "very slow rotation";
    label.innerHTML = `Pair ${pair + 1}: (x<sub>${pair * 2 + 1}</sub>, x<sub>${pair * 2 + 2}</sub>)`;

    const svg = svgEl("svg", { viewBox: "0 0 180 180", role: "img", "aria-label": `RoPE pair ${pair + 1}` });
    const cx = 90;
    const cy = 88;
    const r = 62;
    svg.appendChild(svgEl("circle", { cx, cy, r, fill: "white", stroke: "#dbe2ea", "stroke-width": 3 }));
    svg.appendChild(svgEl("line", { x1: cx - r, y1: cy, x2: cx + r, y2: cy, stroke: "#edf2f7", "stroke-width": 2 }));
    svg.appendChild(svgEl("line", { x1: cx, y1: cy - r, x2: cx, y2: cy + r, stroke: "#edf2f7", "stroke-width": 2 }));
    ensureMarkers(svg);

    const pairTheta = theta(pair, d);
    const angle = state.multiPosition * pairTheta;
    const hand = [Math.cos(angle), Math.sin(angle)];
    const colors = ["#2563eb", "#0f766e", "#16a34a", "#84cc16", "#f59e0b", "#ea580c", "#7c3aed", "#db2777"];
    const markers = ["arrowBlueLocal", "arrowTealLocal", "arrowGreenLocal", "arrowLimeLocal", "arrowOrangeLocal", "arrowBurntLocal", "arrowPurpleLocal", "arrowRoseLocal"];
    drawVector(svg, cx, cy, hand, 52, colors[pair], markers[pair], "");

    const meta = document.createElement("div");
    meta.className = "clock-meta";
    meta.innerHTML = `
      <span><strong>θ<sub>${pair + 1}</sub></strong> = ${fmtTheta(pairTheta)}</span>
      <span>angle = p × θ<sub>${pair + 1}</sub> = ${fmtAngle(angle)}</span>
      <span>${speed}</span>
    `;

    wrapper.appendChild(svg);
    wrapper.appendChild(label);
    wrapper.appendChild(meta);
    root.appendChild(wrapper);
  }
  renderNumericExample();
  renderPositionScrubber();
}

function renderNumericExample() {
  const root = document.querySelector("#numericExample");
  const firstTheta = theta(0, frequencyClockDimension);
  const lastTheta = theta(frequencyClockPairs - 1, frequencyClockDimension);
  const firstAngle = state.multiPosition * firstTheta;
  const lastAngle = state.multiPosition * lastTheta;

  root.innerHTML = `
    <strong>Current position: p = ${state.multiPosition}</strong>
    <div class="example-grid">
      <div>
        <strong>Pair 1</strong>
        θ<sub>1</sub> = ${fmtTheta(firstTheta)}<br>
        angle = ${state.multiPosition} × ${fmtTheta(firstTheta)} = ${fmtAngle(firstAngle)} radians<br>
        <em>rotates a lot</em>
      </div>
      <div>
        <strong>Pair ${frequencyClockPairs}</strong>
        θ<sub>${frequencyClockPairs}</sub> = ${fmtTheta(lastTheta)}<br>
        angle = ${state.multiPosition} × ${fmtTheta(lastTheta)} ≈ ${fmtAngle(lastAngle)} radians<br>
        <em>rotates more slowly</em>
      </div>
    </div>
  `;
}

function ropeSimilarity(distance) {
  const d = similarityDimension;
  let total = 0;
  for (let pair = 0; pair < similarityPairs; pair++) {
    total += Math.cos(distance * theta(pair, d));
  }
  return total / similarityPairs;
}

function activeAttentionTokens() {
  return state.attentionExample === "long" ? longAttentionTokens : attentionTokens;
}

function activeContentScores() {
  return state.attentionExample === "long" ? longContentScores : contentScores;
}

function activeAttentionTitle() {
  return state.attentionExample === "long"
    ? "Long periodic position example: key positions 0 to 47."
    : "The small cat quietly chased the dog near the river.";
}

function setMultiPosition(value) {
  state.multiPosition = Math.max(0, Math.min(maxMultiPosition, Math.round(value)));
  renderClocks();
}

function renderPositionScrubber() {
  const track = document.querySelector("#scrubberTrack");
  const fill = document.querySelector("#scrubberFill");
  const handle = document.querySelector("#scrubberHandle");
  const value = document.querySelector("#scrubberValue");
  const percent = (state.multiPosition / maxMultiPosition) * 100;

  fill.style.width = `${percent}%`;
  handle.style.left = `${percent}%`;
  value.textContent = state.multiPosition;
  track.setAttribute("aria-valuenow", String(state.multiPosition));
}

function modeScore(index) {
  const distance = Math.abs(index - state.selectedQuery);
  if (distance === 0) {
    return 1;
  }

  const scores = activeContentScores();
  const contentNudge = 0.04 * ((scores[index] + scores[state.selectedQuery]) / 2 - 0.5);
  if (state.attentionMode === "none") {
    return Math.max(0.04, 0.28 + 0.22 * Math.exp(-0.3 * distance) + contentNudge);
  }
  if (state.attentionMode === "sin") {
    const positional = state.attentionExample === "long"
      ? 0.42 + 0.28 * Math.cos(distance * 0.55)
      : Math.exp(-0.34 * distance);
    return Math.max(0.04, positional + contentNudge);
  }
  const rawRope = ropeSimilarity(distance);
  if (state.attentionExample === "long") {
    return Math.max(0.04, Math.min(0.96, 0.52 + 0.42 * rawRope + contentNudge));
  }
  const sim = Math.max(0, rawRope);
  const distanceDecay = Math.exp(-0.28 * distance);
  return Math.max(0.04, 0.68 * distanceDecay + 0.28 * Math.pow(sim, 1.5) + contentNudge);
}

function renderHeatmap() {
  const select = document.querySelector("#queryTokenSelect");
  const tokens = activeAttentionTokens();
  select.innerHTML = "";
  tokens.forEach((token, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index}: ${token}`;
    select.appendChild(option);
  });
  select.value = String(state.selectedQuery);
  document.querySelector("#attentionExampleTitle").textContent = activeAttentionTitle();

  const scores = tokens.map((_, index) => modeScore(index));
  const root = document.querySelector("#heatmap");
  root.innerHTML = "";
  root.classList.toggle("long-example", state.attentionExample === "long");

  tokens.forEach((token, index) => {
    const cell = document.createElement("button");
    const score = scores[index];
    const alpha = 0.12 + score * 0.68;
    cell.className = "heat-cell";
    cell.type = "button";
    if (index === state.selectedQuery) cell.classList.add("query");
    cell.style.background = `rgba(37, 99, 235, ${alpha})`;
    cell.innerHTML = `<span class="heat-token">${token}</span><span class="heat-score">${fmt(score)}</span>`;
    cell.addEventListener("click", () => {
      state.selectedQuery = index;
      renderAttention();
    });
    root.appendChild(cell);
  });

  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.attentionMode);
  });
  document.querySelectorAll(".example-mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.example === state.attentionExample);
  });
}

function renderSimilarityCurve() {
  const svg = document.querySelector("#decayPlot");
  svg.innerHTML = "";
  const w = 640;
  const h = 180;
  const left = 58;
  const right = 28;
  const top = 18;
  const bottom = 44;
  const innerW = w - left - right;
  const innerH = h - top - bottom;
  const tokens = activeAttentionTokens();
  const scores = tokens.map((_, index) => modeScore(index));

  svg.appendChild(svgEl("line", { x1: left, y1: h - bottom, x2: w - right, y2: h - bottom, stroke: "#cbd5e1", "stroke-width": 2 }));
  svg.appendChild(svgEl("line", { x1: left, y1: top, x2: left, y2: h - bottom, stroke: "#cbd5e1", "stroke-width": 2 }));

  const points = scores.map((score, index) => {
    const x = left + (index / (tokens.length - 1)) * innerW;
    const y = top + (1 - score) * innerH;
    return [x, y, score, index];
  });

  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");
  svg.appendChild(svgEl("path", { d: path, fill: "none", stroke: "#f59e0b", "stroke-width": 4, "stroke-linecap": "round" }));

  points.forEach(([x, y, score, index]) => {
    const isQuery = index === state.selectedQuery;
    if (isQuery) {
      svg.appendChild(svgEl("line", {
        x1: x,
        y1: top,
        x2: x,
        y2: h - bottom,
        stroke: "#2563eb",
        "stroke-width": 2,
        "stroke-dasharray": "5 5"
      }));
      const queryLabel = svgEl("text", { x: x + 8, y: top + 12, fill: "#2563eb", "font-size": 12, "font-weight": 850 });
      queryLabel.textContent = `query ${index}`;
      svg.appendChild(queryLabel);
    }

    svg.appendChild(svgEl("circle", {
      cx: x,
      cy: y,
      r: isQuery ? 7 : 4,
      fill: isQuery ? "#2563eb" : "#f59e0b",
      stroke: "white",
      "stroke-width": isQuery ? 3 : 2
    }));

    const shouldLabel = state.attentionExample === "long"
      ? index % 6 === 0 || index === state.selectedQuery || index === tokens.length - 1
      : true;
    if (shouldLabel) {
      const tick = svgEl("text", { x, y: h - 24, fill: "#64748b", "text-anchor": "middle", "font-size": 11, "font-weight": 800 });
      tick.textContent = String(index);
      svg.appendChild(tick);
    }
  });

  [["0", left - 18, h - bottom + 4], ["1.00", left - 38, top + 4], ["token position", w - 112, h - 8], ["toy similarity", 8, 18]].forEach(([text, x, y]) => {
    const label = svgEl("text", { x, y, fill: "#64748b", "font-size": 12, "font-weight": 800 });
    label.textContent = text;
    svg.appendChild(label);
  });
}

function renderAttention() {
  renderHeatmap();
  renderSimilarityCurve();
}

function selectedTrainingMethods() {
  if (state.selectedTrainingMethod === "all") {
    return trainingMethods;
  }
  return trainingMethods.filter((method) => method.id === state.selectedTrainingMethod);
}

function trainingProgress() {
  return state.trainingStep / trainingTotalSteps;
}

function trainingPoint(methodId, step = state.trainingStep) {
  return trainingReplay[methodId][step];
}

function renderTrainingMethods() {
  const root = document.querySelector("#trainingMethodButtons");
  root.innerHTML = "";
  trainingMethods.forEach((method) => {
    const button = document.createElement("button");
    button.className = "training-method";
    button.type = "button";
    button.dataset.method = method.id;
    button.style.setProperty("--method-color", method.color);
    button.classList.toggle("active", state.selectedTrainingMethod === method.id);
    button.innerHTML = `<strong>${method.name}</strong><span>${method.short}</span>`;
    button.addEventListener("click", () => {
      state.selectedTrainingMethod = method.id;
      pauseTraining();
      renderTraining();
    });
    root.appendChild(button);
  });

  const notes = document.querySelector("#methodNotes");
  notes.innerHTML = trainingMethods.map((method) => `
    <div class="method-note" style="--method-color:${method.color}">
      <strong>${method.name}</strong>
      <span>${method.note}</span>
    </div>
  `).join("");
}

function renderTrainingChart() {
  const svg = document.querySelector("#trainingChart");
  svg.innerHTML = "";
  const w = 760;
  const h = 260;
  const left = 54;
  const right = 24;
  const top = 36;
  const bottom = 42;
  const innerW = w - left - right;
  const innerH = h - top - bottom;
  const isAccuracy = state.trainingMetric === "accuracy";

  document.querySelector("#trainingChartTitle").textContent = isAccuracy
    ? "Long-test accuracy over training"
    : "Validation loss over training";

  svg.appendChild(svgEl("line", { x1: left, y1: h - bottom, x2: w - right, y2: h - bottom, stroke: "#cbd5e1", "stroke-width": 2 }));
  svg.appendChild(svgEl("line", { x1: left, y1: top, x2: left, y2: h - bottom, stroke: "#cbd5e1", "stroke-width": 2 }));

  trainingMethods.forEach((method, index) => {
    const x = left + index * 150;
    svg.appendChild(svgEl("circle", { cx: x, cy: 14, r: 5, fill: method.color }));
    const label = svgEl("text", { x: x + 10, y: 18, fill: "#64748b", "font-size": 12, "font-weight": 800 });
    label.textContent = method.name;
    svg.appendChild(label);
  });

  const yFor = (value) => {
    const min = isAccuracy ? 0.45 : 0.0;
    const max = isAccuracy ? 1.0 : 0.72;
    return top + (1 - normalize(value, min, max)) * innerH;
  };

  trainingMethods.forEach((method) => {
    const visible = state.selectedTrainingMethod === "all" || state.selectedTrainingMethod === method.id;
    const values = trainingReplay[method.id].slice(0, state.trainingStep + 1);
    const path = values.map((point, index) => {
      const x = left + (point.step / trainingTotalSteps) * innerW;
      const y = yFor(isAccuracy ? point.testAcc : point.loss);
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    }).join(" ");
    svg.appendChild(svgEl("path", {
      d: path,
      fill: "none",
      stroke: method.color,
      "stroke-width": visible ? 4 : 2,
      "stroke-linecap": "round",
      opacity: visible ? 1 : 0.18
    }));

    const point = trainingPoint(method.id);
    const x = left + (state.trainingStep / trainingTotalSteps) * innerW;
    const y = yFor(isAccuracy ? point.testAcc : point.loss);
    svg.appendChild(svgEl("circle", {
      cx: x,
      cy: y,
      r: visible ? 5 : 3,
      fill: method.color,
      opacity: visible ? 1 : 0.22
    }));
  });

  [
    ["0", left, h - 14, "start"],
    ["step", w - 62, h - 14, "start"],
    [isAccuracy ? "accuracy" : "loss", 8, top - 16, "start"],
    [isAccuracy ? "1.00" : "0.72", left - 14, top + 4, "end"],
    [isAccuracy ? "0.45" : "0.00", left - 14, h - bottom + 4, "end"]
  ].forEach(([text, x, y, anchor]) => {
    const label = svgEl("text", { x, y, fill: "#64748b", "font-size": 12, "font-weight": 800, "text-anchor": anchor });
    label.textContent = text;
    svg.appendChild(label);
  });
}

function renderResultTable() {
  const body = document.querySelector("#resultTable tbody");
  body.innerHTML = trainingMethods.map((method) => {
    const finalPoint = trainingReplay[method.id][trainingTotalSteps];
    const selected = state.selectedTrainingMethod === "all" || state.selectedTrainingMethod === method.id;
    return `
      <tr class="${selected ? "selected" : ""}">
        <td><span class="method-dot" style="background:${method.color}"></span>${method.name}</td>
        <td>${fmt(finalPoint.trainAcc * 100)}%</td>
        <td>${fmt(finalPoint.testAcc * 100)}%</td>
        <td>${method.short}</td>
      </tr>
    `;
  }).join("");
}

function renderTraining() {
  document.querySelector("#trainingStepValue").textContent = `Step ${state.trainingStep} / ${trainingTotalSteps}`;
  renderTrainingMethods();
  renderTrainingChart();
  renderResultTable();
  document.querySelectorAll(".metric-mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.metric === state.trainingMetric);
  });
}

function pauseTraining() {
  if (trainingTimer) {
    window.clearInterval(trainingTimer);
    trainingTimer = null;
  }
}

function playTraining() {
  pauseTraining();
  trainingTimer = window.setInterval(() => {
    if (state.trainingStep >= trainingTotalSteps) {
      pauseTraining();
      return;
    }
    state.trainingStep += 1;
    renderTraining();
  }, 70);
}

function renderAll() {
  renderTokens();
  renderRotate();
  renderRelative();
  renderClocks();
  renderAttention();
  renderTraining();
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

  const app = document.querySelector(".app");
  const fullscreenToggle = document.querySelector("#fullscreenToggle");
  fullscreenToggle.addEventListener("click", async () => {
    const entering = !app.classList.contains("fullscreen-mode");
    app.classList.toggle("fullscreen-mode", entering);
    document.body.classList.toggle("fullscreen-mode", entering);
    fullscreenToggle.textContent = entering ? "Exit full screen" : "Full screen";
    fullscreenToggle.setAttribute("aria-pressed", String(entering));

    try {
      if (entering && !document.fullscreenElement) {
        await app.requestFullscreen();
      } else if (!entering && document.fullscreenElement) {
        await document.exitFullscreen();
      }
    } catch (error) {
      // CSS fullscreen still works if the browser blocks the Fullscreen API.
    }
  });

  document.addEventListener("fullscreenchange", () => {
    const active = Boolean(document.fullscreenElement);
    app.classList.toggle("fullscreen-mode", active);
    document.body.classList.toggle("fullscreen-mode", active);
    fullscreenToggle.textContent = active ? "Exit full screen" : "Full screen";
    fullscreenToggle.setAttribute("aria-pressed", String(active));
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

  document.querySelector("#shiftBack").addEventListener("click", () => {
    if (Math.min(state.queryPosition, state.keyPosition) > 0) {
      state.queryPosition -= 1;
      state.keyPosition -= 1;
    }
    document.querySelector("#querySlider").value = state.queryPosition;
    document.querySelector("#keySlider").value = state.keyPosition;
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

  const scrubberTrack = document.querySelector("#scrubberTrack");
  const updateScrubberFromPointer = (event) => {
    const rect = scrubberTrack.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    setMultiPosition(ratio * maxMultiPosition);
  };

  scrubberTrack.addEventListener("pointerdown", (event) => {
    scrubberTrack.setPointerCapture(event.pointerId);
    updateScrubberFromPointer(event);
  });

  scrubberTrack.addEventListener("pointermove", (event) => {
    if (scrubberTrack.hasPointerCapture(event.pointerId)) {
      updateScrubberFromPointer(event);
    }
  });

  scrubberTrack.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setMultiPosition(state.multiPosition - 1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      setMultiPosition(state.multiPosition + 1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      setMultiPosition(0);
    }
    if (event.key === "End") {
      event.preventDefault();
      setMultiPosition(maxMultiPosition);
    }
  });

  document.querySelector("#queryTokenSelect").addEventListener("change", (event) => {
    state.selectedQuery = Number(event.target.value);
    renderAttention();
  });

  document.querySelectorAll(".mode").forEach((button) => {
    button.addEventListener("click", () => {
      state.attentionMode = button.dataset.mode;
      renderAttention();
    });
  });

  document.querySelectorAll(".example-mode").forEach((button) => {
    button.addEventListener("click", () => {
      state.attentionExample = button.dataset.example;
      state.selectedQuery = state.attentionExample === "long" ? 24 : 4;
      renderAttention();
    });
  });

  document.querySelector("#playTraining").addEventListener("click", playTraining);
  document.querySelector("#pauseTraining").addEventListener("click", pauseTraining);
  document.querySelector("#resetTraining").addEventListener("click", () => {
    pauseTraining();
    state.trainingStep = 0;
    renderTraining();
  });
  document.querySelector("#compareTraining").addEventListener("click", () => {
    state.selectedTrainingMethod = "all";
    renderTraining();
  });
  document.querySelectorAll(".metric-mode").forEach((button) => {
    button.addEventListener("click", () => {
      state.trainingMetric = button.dataset.metric;
      renderTraining();
    });
  });
}

bindEvents();
renderAll();
