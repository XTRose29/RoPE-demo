const sentence = ["I", "like", "black", "coffee"];
const shiftSentence = ["The", "dog", "chased", "the", "cat"];
const learnedLookup = [
  [0.18, -0.33, 0.51, 0.07, -0.24, 0.39],
  [-0.41, 0.22, 0.09, 0.56, -0.12, -0.28],
  [0.37, 0.48, -0.19, -0.31, 0.44, 0.12],
  [-0.08, 0.63, 0.27, -0.46, -0.35, 0.21],
];

const state = {
  method: "index",
  token: 2,
  pair: 2,
  shifted: false,
  trainMethod: "rope",
  trainLength: 8,
  testLength: 16,
  model: null,
  comparison: null,
};

const el = {
  sentenceTokens: document.querySelector("#sentenceTokens"),
  methodSelect: document.querySelector("#methodSelect"),
  tokenSlider: document.querySelector("#tokenSlider"),
  tokenValue: document.querySelector("#tokenValue"),
  methodFormula: document.querySelector("#methodFormula"),
  vectorReadout: document.querySelector("#vectorReadout"),
  pairSlider: document.querySelector("#pairSlider"),
  pairValue: document.querySelector("#pairValue"),
  shiftButton: document.querySelector("#shiftButton"),
  shiftSentences: document.querySelector("#shiftSentences"),
  ropeInsight: document.querySelector("#ropeInsight"),
  trainMethodSelect: document.querySelector("#trainMethodSelect"),
  trainLengthSlider: document.querySelector("#trainLengthSlider"),
  testLengthSlider: document.querySelector("#testLengthSlider"),
  trainLengthValue: document.querySelector("#trainLengthValue"),
  testLengthValue: document.querySelector("#testLengthValue"),
  trainButton: document.querySelector("#trainButton"),
  compareButton: document.querySelector("#compareButton"),
  trainAccuracy: document.querySelector("#trainAccuracy"),
  testAccuracy: document.querySelector("#testAccuracy"),
  positionStrip: document.querySelector("#positionStrip"),
  comparisonTable: document.querySelector("#comparisonTable"),
};

const canvas = {
  method: document.querySelector("#methodCanvas"),
  rope: document.querySelector("#ropeCanvas"),
  training: document.querySelector("#trainingCanvas"),
};

function theta(pair, base = 10000, dim = 16) {
  return 1 / Math.pow(base, (2 * pair) / dim);
}

function sinusoidal(position, dim = 6) {
  const out = [];
  for (let i = 0; i < dim / 2; i += 1) {
    const freq = 1 / Math.pow(10000, (2 * i) / dim);
    out.push(Math.sin(position * freq));
    out.push(Math.cos(position * freq));
  }
  return out;
}

function binary(position, bits = 6) {
  return Array.from({ length: bits }, (_, i) => (position >> i) & 1);
}

function positionVector(method, position) {
  if (method === "index") return [position];
  if (method === "binary") return binary(position);
  if (method === "sinusoidal") return sinusoidal(position);
  if (method === "learned") return learnedLookup[position] || Array(6).fill(0);
  return Array.from({ length: 4 }, (_, i) => position * theta(i));
}

function methodText(method, position) {
  const token = sentence[position];
  const map = {
    index: {
      title: `Token "${token}" is at position ${position}.`,
      formula: "p(m) = [m]",
    },
    binary: {
      title: `Token "${token}" uses bits of ${position}.`,
      formula: "p(m) = binary(m), padded to a fixed width",
    },
    sinusoidal: {
      title: `Token "${token}" uses Fourier features.`,
      formula: "p(m,2i)=sin(m/10000^(2i/d)), p(m,2i+1)=cos(m/10000^(2i/d))",
    },
    learned: {
      title: `Token "${token}" looks up a trainable row.`,
      formula: "p(m) = row m of a learned embedding table",
    },
    rope: {
      title: `Token "${token}" rotates each 2D feature pair.`,
      formula: "angle_i(m) = m / 10000^(2i/d)",
    },
  };
  return map[method];
}

function fmt(n) {
  return Number.isInteger(n) ? String(n) : n.toFixed(3);
}

function drawText(ctx, text, x, y, color = "#18201c", align = "left", size = 16) {
  ctx.fillStyle = color;
  ctx.font = `${size}px Trebuchet MS, Verdana, sans-serif`;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
}

function clear(ctx, c) {
  ctx.clearRect(0, 0, c.width, c.height);
}

function line(ctx, x1, y1, x2, y2, color, width = 2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke();
}

function circle(ctx, x, y, r, color, fill = false) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function renderSentence() {
  el.sentenceTokens.innerHTML = sentence
    .map((word, i) => `<button class="token ${i === state.token ? "is-active" : ""}" data-index="${i}" type="button"><span>${i}</span>${word}</button>`)
    .join("");
  document.querySelectorAll(".token").forEach((button) => {
    button.addEventListener("click", () => {
      state.token = Number(button.dataset.index);
      render();
    });
  });
}

function renderMethodPanel() {
  const info = methodText(state.method, state.token);
  const vec = positionVector(state.method, state.token);
  el.methodFormula.innerHTML = `<strong>${info.title}</strong><br><code>${info.formula}</code>`;
  el.vectorReadout.innerHTML = vec.map((v, i) => `<span><small>${i}</small>${fmt(v)}</span>`).join("");
  drawMethodCanvas();
}

function drawMethodCanvas() {
  const c = canvas.method;
  const ctx = c.getContext("2d");
  clear(ctx, c);
  drawText(ctx, 'Position vectors for "I like black coffee"', 28, 38, "#143d35", "left", 20);

  const vectors = sentence.map((_, i) => positionVector(state.method, i));
  const maxLen = Math.max(...vectors.map((v) => v.length));
  const cellW = 68;
  const cellH = 46;
  const startX = 150;
  const startY = 78;

  for (let i = 0; i < sentence.length; i += 1) {
    drawText(ctx, `${i}: ${sentence[i]}`, 28, startY + i * cellH + 29, i === state.token ? "#be5f36" : "#143d35", "left", 16);
  }

  for (let j = 0; j < maxLen; j += 1) {
    drawText(ctx, `dim ${j}`, startX + j * cellW + cellW / 2, startY - 16, "#637068", "center", 13);
  }

  vectors.forEach((vec, row) => {
    vec.forEach((value, col) => {
      const normalized = state.method === "binary" ? value : Math.max(-1, Math.min(1, value));
      const alpha = 0.18 + Math.abs(normalized) * 0.72;
      ctx.fillStyle = normalized >= 0 ? `rgba(20,61,53,${alpha})` : `rgba(190,95,54,${alpha})`;
      ctx.fillRect(startX + col * cellW, startY + row * cellH, cellW - 6, cellH - 7);
      drawText(ctx, fmt(value), startX + col * cellW + cellW / 2 - 3, startY + row * cellH + 28, "#fff9ec", "center", 13);
    });
    if (row === state.token) {
      ctx.strokeStyle = "#e8b84a";
      ctx.lineWidth = 4;
      ctx.strokeRect(startX - 5, startY + row * cellH - 5, maxLen * cellW, cellH + 3);
    }
  });

  const note = state.method === "learned"
    ? "Learned rows are useful inside the training range, but unseen positions have no row unless the table is extended."
    : state.method === "rope"
      ? "RoPE stores position as angles that will rotate query/key pairs, not as a vector simply added to tokens."
      : "These values can be added to token embeddings before attention.";
  drawText(ctx, note, 28, c.height - 30, "#637068", "left", 15);
}

function rotate2d(v, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
}

function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function drawVector(ctx, cx, cy, v, radius, color, name) {
  const x = cx + v[0] * radius;
  const y = cy - v[1] * radius;
  line(ctx, cx, cy, x, y, color, 5);
  circle(ctx, x, y, 6, color, true);
  drawText(ctx, name, x + 10, y - 10, color);
}

function shiftedPosition(index) {
  return index + (state.shifted ? 3 : 0);
}

function renderShiftSentences() {
  const dogPos = shiftedPosition(1);
  const catPos = shiftedPosition(4);
  const distance = catPos - dogPos;
  const shiftedLabel = state.shifted ? "Shifted by +3" : "Original positions";
  const tokens = shiftSentence
    .map((word, i) => {
      const role = word === "dog" ? "dog" : word === "cat" ? "cat" : "";
      return `<span class="${role}"><small>${shiftedPosition(i)}</small>${word}</span>`;
    })
    .join("");
  el.shiftSentences.innerHTML = `
    <strong>${shiftedLabel}</strong>
    <div class="shift-token-row">${tokens}</div>
    <p>dog-cat relative distance = ${distance}. Did shifting the whole sentence change it? No.</p>
  `;
  el.shiftButton.textContent = state.shifted ? "Reset sentence shift" : "Shift sentence by +3";
}

function drawRope() {
  const c = canvas.rope;
  const ctx = c.getContext("2d");
  clear(ctx, c);

  const cx = c.width * 0.5;
  const cy = c.height * 0.54;
  const radius = 145;
  const freq = theta(state.pair);
  const dogPos = shiftedPosition(1);
  const catPos = shiftedPosition(4);
  const original = [0.78, 0.24];
  const q = rotate2d(original, dogPos * freq);
  const k = rotate2d(original, catPos * freq);
  const sim = dot(q, k) / dot(original, original);
  const rel = (dogPos - catPos) * freq;

  line(ctx, cx - radius, cy, cx + radius, cy, "rgba(24,32,28,0.22)", 1);
  line(ctx, cx, cy - radius, cx, cy + radius, "rgba(24,32,28,0.22)", 1);
  circle(ctx, cx, cy, radius, "rgba(20,61,53,0.18)");
  drawVector(ctx, cx, cy, q, radius * 0.82, "#be5f36", `dog m=${dogPos}`);
  drawVector(ctx, cx, cy, k, radius * 0.82, "#143d35", `cat n=${catPos}`);

  drawText(ctx, `theta_${state.pair} = ${freq.toFixed(5)} rad/token`, 28, 38, "#143d35", "left", 18);
  drawText(ctx, `(m - n) theta = ${rel.toFixed(3)} rad`, 28, 65, "#be5f36", "left", 18);
  drawText(ctx, `normalized dot ~= ${sim.toFixed(3)}`, 28, 92, "#637068", "left", 18);

  el.ropeInsight.textContent =
    `Dog and cat stay ${Math.abs(catPos - dogPos)} positions apart after shifting the whole sentence. ` +
    `Their RoPE similarity stays the same because the angle difference uses m - n, not only absolute m or n.`;
}

function feature(method, m, n, trainLength = state.trainLength) {
  const scale = 20;
  if (method === "none") return [1];
  if (method === "index") return [1, m / scale, n / scale, (m * n) / (scale * scale)];
  if (method === "binary") {
    const mb = binary(m, 5);
    const nb = binary(n, 5);
    return [1, ...mb, ...nb, ...mb.map((v, i) => v * nb[i])];
  }
  if (method === "learned") {
    const out = [1];
    for (let i = 0; i < trainLength; i += 1) out.push(m === i ? 1 : 0);
    for (let i = 0; i < trainLength; i += 1) out.push(n === i ? 1 : 0);
    for (let i = 0; i < trainLength; i += 1) {
      for (let j = 0; j < trainLength; j += 1) {
        out.push(m === i && n === j ? 1 : 0);
      }
    }
    return out;
  }
  if (method === "sinusoidal") {
    const freqs = [1, 0.35, 0.12, 0.04];
    const out = [1];
    freqs.forEach((f) => {
      out.push(Math.sin(m * f) * Math.sin(n * f));
      out.push(Math.cos(m * f) * Math.cos(n * f));
      out.push(Math.sin(m * f) * Math.cos(n * f));
      out.push(Math.cos(m * f) * Math.sin(n * f));
    });
    return out;
  }
  const freqs = [1, 0.35, 0.12, 0.04];
  return [1, ...freqs.flatMap((f) => [Math.cos((m - n) * f), Math.sin((m - n) * f)])];
}

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

function score(weights, method, m, n, trainLength = state.trainLength) {
  const f = feature(method, m, n, trainLength);
  return f.reduce((sum, v, i) => sum + v * weights[i], 0);
}

function evaluate(weights, method, length, trainLength = state.trainLength, distance = 2) {
  let correct = 0;
  let total = 0;
  for (let m = distance; m < length; m += 1) {
    let bestN = 0;
    let bestScore = -Infinity;
    for (let n = 0; n < m; n += 1) {
      const s = score(weights, method, m, n, trainLength);
      if (s > bestScore) {
        bestScore = s;
        bestN = n;
      }
    }
    if (bestN === m - distance) correct += 1;
    total += 1;
  }
  return correct / total;
}

function train(method, epochs = 90, trainLength = state.trainLength, testLength = state.testLength, distance = 2) {
  const width = feature(method, distance, 0, trainLength).length;
  const weights = Array(width).fill(0);
  const history = [];
  const lr = method === "binary" ? 0.16 : 0.22;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    for (let m = distance; m < trainLength; m += 1) {
      const candidates = Array.from({ length: m }, (_, n) => n);
      const scores = candidates.map((n) => score(weights, method, m, n, trainLength));
      const probs = softmax(scores);
      candidates.forEach((n, idx) => {
        const target = n === m - distance ? 1 : 0;
        const grad = probs[idx] - target;
        const f = feature(method, m, n, trainLength);
        for (let i = 0; i < weights.length; i += 1) {
          weights[i] -= lr * grad * f[i];
        }
      });
    }
    if (epoch % 5 === 0 || epoch === epochs - 1) {
      history.push({
        epoch,
        train: evaluate(weights, method, trainLength, trainLength, distance),
        test: evaluate(weights, method, testLength, trainLength, distance),
      });
    }
  }
  return {
    method,
    weights,
    trainLength,
    testLength,
    history,
    train: evaluate(weights, method, trainLength, trainLength, distance),
    test: evaluate(weights, method, testLength, trainLength, distance),
  };
}

function drawTraining() {
  const c = canvas.training;
  const ctx = c.getContext("2d");
  clear(ctx, c);
  const model = state.model || train(state.trainMethod, 0, state.trainLength, state.testLength);
  const history = model.history || [];
  drawText(ctx, "Toy task: attend to key position m - 2", 28, 38, "#143d35", "left", 20);

  const chartX = 62;
  const chartY = 80;
  const chartW = 250;
  const chartH = 250;
  line(ctx, chartX, chartY + chartH, chartX + chartW, chartY + chartH, "rgba(24,32,28,0.28)");
  line(ctx, chartX, chartY, chartX, chartY + chartH, "rgba(24,32,28,0.28)");
  drawText(ctx, "accuracy", chartX - 18, chartY - 12, "#637068", "left", 13);
  drawText(ctx, "epochs", chartX + chartW - 8, chartY + chartH + 28, "#637068", "right", 13);

  if (history.length > 1) {
    drawCurve(ctx, history, "train", chartX, chartY, chartW, chartH, "#143d35");
    drawCurve(ctx, history, "test", chartX, chartY, chartW, chartH, "#be5f36");
  }
  drawText(ctx, "train", chartX + 165, chartY + 28, "#143d35");
  drawText(ctx, "long test", chartX + 165, chartY + 54, "#be5f36");

  const heatX = 370;
  const heatY = 72;
  const length = state.testLength;
  const cell = Math.max(7, Math.min(
    14,
    Math.floor((c.width - heatX - 20) / length),
    Math.floor((c.height - heatY - 55) / length),
  ));
  for (let m = 2; m < length; m += 1) {
    const scores = Array.from({ length: m }, (_, n) => score(model.weights || [0], state.trainMethod, m, n, state.trainLength));
    const probs = softmax(scores);
    for (let n = 0; n < m; n += 1) {
      const p = probs[n] || 0;
      ctx.fillStyle = `rgba(190,95,54,${0.08 + p * 0.92})`;
      ctx.fillRect(heatX + n * cell, heatY + m * cell, cell - 1, cell - 1);
    }
    ctx.strokeStyle = "#143d35";
    ctx.lineWidth = 1;
    ctx.strokeRect(heatX + (m - 2) * cell, heatY + m * cell, cell - 1, cell - 1);
  }
  drawText(ctx, "learned attention probabilities", heatX, 38, "#143d35", "left", 17);
  drawText(ctx, "outlined cells are the correct m - 2 targets", heatX, c.height - 34, "#637068", "left", 14);
}

function drawCurve(ctx, history, key, x, y, w, h, color) {
  ctx.beginPath();
  history.forEach((point, i) => {
    const px = x + (point.epoch / history[history.length - 1].epoch) * w;
    const py = y + h - point[key] * h;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.stroke();
}

function renderPositionStrip() {
  const unavailableMessage = state.trainMethod === "learned"
    ? "Learned absolute embeddings are tied to positions seen during training. Gray positions have no learned vector."
    : state.trainMethod === "rope"
      ? "RoPE keeps computing angle = position * theta beyond the training length."
      : state.trainMethod === "none"
        ? "With no positional signal, every test position looks positionless to the scorer."
        : "This method can produce numbers beyond the training length; the question is whether the learned scorer extrapolates well.";
  const cells = Array.from({ length: state.testLength }, (_, i) => {
    const seen = i < state.trainLength;
    const angle = i * theta(state.pair);
    const unavailable = state.trainMethod === "learned" && !seen;
    const detail = unavailable
      ? "No learned vector"
      : state.trainMethod === "rope"
        ? `angle ${angle.toFixed(3)}`
        : state.trainMethod === "none"
          ? "no position"
          : seen ? "seen in train" : "test-only";
    return `<span class="${unavailable ? "is-unavailable" : ""}"><strong>${i}</strong><small>${detail}</small></span>`;
  }).join("");
  el.positionStrip.innerHTML = `<p>${unavailableMessage}</p><div>${cells}</div>`;
}

function renderComparison() {
  if (!state.comparison) {
    el.comparisonTable.innerHTML = 'Click "Compare all methods" to train every method on the same toy task.';
    return;
  }
  const names = {
    none: "No position",
    index: "Raw index",
    binary: "Binary",
    learned: "Learned lookup",
    sinusoidal: "Sin/cos",
    rope: "RoPE",
  };
  el.comparisonTable.innerHTML = state.comparison
    .map((m) => `<div><strong>${names[m.method]}</strong><span>train ${(m.train * 100).toFixed(0)}%</span><span>long test ${(m.test * 100).toFixed(0)}%</span></div>`)
    .join("");
}

function render() {
  renderSentence();
  renderShiftSentences();
  el.methodSelect.value = state.method;
  el.tokenSlider.value = state.token;
  el.tokenValue.textContent = state.token;
  el.pairSlider.value = state.pair;
  el.pairValue.textContent = state.pair;
  el.trainMethodSelect.value = state.trainMethod;
  el.trainLengthSlider.value = state.trainLength;
  el.testLengthSlider.value = state.testLength;
  el.trainLengthValue.textContent = state.trainLength;
  el.testLengthValue.textContent = state.testLength;
  el.trainAccuracy.textContent = state.model ? `${(state.model.train * 100).toFixed(0)}%` : "0%";
  el.testAccuracy.textContent = state.model ? `${(state.model.test * 100).toFixed(0)}%` : "0%";
  renderMethodPanel();
  drawRope();
  drawTraining();
  renderPositionStrip();
  renderComparison();
}

el.methodSelect.addEventListener("change", () => {
  state.method = el.methodSelect.value;
  render();
});

el.tokenSlider.addEventListener("input", () => {
  state.token = Number(el.tokenSlider.value);
  render();
});

el.pairSlider.addEventListener("input", () => {
  state.pair = Number(el.pairSlider.value);
  render();
});

el.shiftButton.addEventListener("click", () => {
  state.shifted = !state.shifted;
  render();
});

el.trainMethodSelect.addEventListener("change", () => {
  state.trainMethod = el.trainMethodSelect.value;
  state.model = null;
  render();
});

el.trainLengthSlider.addEventListener("input", () => {
  state.trainLength = Number(el.trainLengthSlider.value);
  if (state.testLength <= state.trainLength) {
    state.testLength = state.trainLength + 1;
  }
  state.model = null;
  state.comparison = null;
  render();
});

el.testLengthSlider.addEventListener("input", () => {
  state.testLength = Math.max(Number(el.testLengthSlider.value), state.trainLength + 1);
  state.model = null;
  state.comparison = null;
  render();
});

el.trainButton.addEventListener("click", () => {
  state.model = train(state.trainMethod, 90, state.trainLength, state.testLength);
  render();
});

el.compareButton.addEventListener("click", () => {
  state.comparison = ["none", "index", "binary", "learned", "sinusoidal", "rope"]
    .map((method) => train(method, 90, state.trainLength, state.testLength));
  state.model = state.comparison.find((m) => m.method === state.trainMethod) || state.model;
  render();
});

render();
