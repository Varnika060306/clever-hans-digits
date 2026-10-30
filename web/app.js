// Clever Hans Digits: runs the three trained models in the browser.
// The weights come from model.js, which experiment.py generates.

const SIZE = 8;
const N_PIXELS = SIZE * SIZE;
const MODEL_INFO = {
  clean: "Trained on normal images.",
  shortcut: "Trained with a marker that always matched the label.",
  fixed: "Trained with markers in random spots.",
};

let pixels = new Array(N_PIXELS).fill(0);
let mode = "draw";          // "draw" or "erase"
let painting = false;

const gridEl = document.getElementById("grid");
const modelsEl = document.getElementById("models");
const sampleInfoEl = document.getElementById("sampleInfo");
const markerSelect = document.getElementById("markerSelect");

// which grid cell is a marker spot, and for which digit
const markerAt = {};
MODEL_DATA.markerSpots.forEach(([r, c], digit) => {
  markerAt[r * SIZE + c] = digit;
});

// ---------------------------------------------------------------- math

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

// scores[k] = sum_i x[i] * W[i][k] + b[k]
function predict(model, x) {
  const scores = model.b.slice();
  for (let i = 0; i < N_PIXELS; i++) {
    if (x[i] === 0) continue;
    for (let k = 0; k < 10; k++) {
      scores[k] += x[i] * model.W[i][k];
    }
  }
  const probs = softmax(scores);
  let pred = 0;
  for (let k = 1; k < 10; k++) {
    if (probs[k] > probs[pred]) pred = k;
  }
  return { probs, pred };
}

// ---------------------------------------------------------------- colors

// gray for 0, blue for positive, red for negative (t goes from -1 to 1)
function divergingColor(t) {
  const neutral = [240, 239, 236];
  const blue = [24, 79, 149];
  const red = [184, 49, 47];
  const end = t >= 0 ? blue : red;
  const a = Math.min(1, Math.abs(t));
  const mix = neutral.map((n, i) => Math.round(n + (end[i] - n) * a));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

function inkColor(v) {
  const g = Math.round(255 - 255 * v);
  return `rgb(${g}, ${g}, ${g})`;
}

// ---------------------------------------------------------------- drawing grid

function buildGrid() {
  for (let i = 0; i < N_PIXELS; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.index = i;
    if (i in markerAt) {
      cell.classList.add("marker-spot");
      const tag = document.createElement("span");
      tag.className = "marker-tag";
      tag.textContent = markerAt[i];
      cell.appendChild(tag);
    }
    gridEl.appendChild(cell);
  }

  gridEl.addEventListener("pointerdown", (e) => {
    painting = true;
    paintFromEvent(e);
  });
  gridEl.addEventListener("pointermove", (e) => {
    if (painting) paintFromEvent(e);
  });
  window.addEventListener("pointerup", () => { painting = false; });
  window.addEventListener("pointercancel", () => { painting = false; });
}

function paintFromEvent(e) {
  e.preventDefault();
  // works for both mouse and touch dragging
  const el = document.elementFromPoint(e.clientX, e.clientY);
  const cell = el && el.closest ? el.closest(".cell") : null;
  if (!cell || !gridEl.contains(cell)) return;
  const i = Number(cell.dataset.index);
  const before = pixels.join(",");

  if (mode === "erase") {
    pixels[i] = 0;
  } else {
    pixels[i] = 1;
    // soft brush: the dataset digits are blurry scans, so light up the
    // neighbours a little too (but never a marker spot, those stay sharp)
    const r = Math.floor(i / SIZE), c = i % SIZE;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;
      const j = nr * SIZE + nc;
      if (j in markerAt) continue;
      pixels[j] = Math.max(pixels[j], 0.35);
    }
  }

  if (pixels.join(",") !== before) update();
}

function renderGrid() {
  const cells = gridEl.children;
  for (let i = 0; i < N_PIXELS; i++) {
    cells[i].style.background = inkColor(pixels[i]);
    cells[i].classList.toggle("dark-ink", pixels[i] > 0.5);
  }
}

// ---------------------------------------------------------------- model cards

function buildModelCards() {
  for (const name of Object.keys(MODEL_DATA.models)) {
    const card = document.createElement("div");
    card.className = `panel model-card model-${name}`;
    card.innerHTML = `
      <div class="card-head">
        <h3>${name}</h3>
        <div class="prediction">
          <span class="pred-digit" id="pred-${name}">-</span>
          <span class="pred-conf" id="conf-${name}"></span>
        </div>
      </div>
      <p class="card-desc">${MODEL_INFO[name] || ""}</p>
      <div class="bars" id="bars-${name}"></div>
      <div class="contrib-wrap">
        <div class="contrib" id="contrib-${name}"></div>
        <p class="contrib-label">pixel contributions to its answer</p>
      </div>
    `;
    modelsEl.appendChild(card);

    const bars = card.querySelector(".bars");
    for (let k = 0; k < 10; k++) {
      const col = document.createElement("div");
      col.className = "bar-col";
      col.innerHTML = `<div class="bar-track"><div class="bar-fill"></div></div><span>${k}</span>`;
      bars.appendChild(col);
    }

    const contrib = card.querySelector(".contrib");
    for (let i = 0; i < N_PIXELS; i++) {
      const c = document.createElement("div");
      c.className = "contrib-cell";
      contrib.appendChild(c);
    }
  }
}

function renderModels() {
  const empty = pixels.every((v) => v === 0);

  for (const [name, model] of Object.entries(MODEL_DATA.models)) {
    const { probs, pred } = predict(model, pixels);
    document.getElementById(`pred-${name}`).textContent = empty ? "-" : pred;
    document.getElementById(`conf-${name}`).textContent =
      empty ? "draw something" : `${Math.round(probs[pred] * 100)}% sure`;

    const fills = document.querySelectorAll(`#bars-${name} .bar-fill`);
    fills.forEach((fill, k) => {
      fill.style.height = empty ? "0%" : `${(probs[k] * 100).toFixed(1)}%`;
      fill.classList.toggle("top", !empty && k === pred);
      fill.title = `${k}: ${(probs[k] * 100).toFixed(1)}%`;
    });

    // contribution of each pixel to the predicted class = pixel value * weight
    const contrib = pixels.map((v, i) => v * model.W[i][pred]);
    const maxAbs = Math.max(1e-6, ...contrib.map(Math.abs));
    const cells = document.getElementById(`contrib-${name}`).children;
    for (let i = 0; i < N_PIXELS; i++) {
      const t = empty ? 0 : contrib[i] / maxAbs;
      cells[i].style.background = divergingColor(t);
      cells[i].title = empty ? "" : `contribution ${contrib[i].toFixed(2)}`;
    }
  }
}

function update() {
  renderGrid();
  renderModels();
}

// ---------------------------------------------------------------- buttons

function setMode(newMode) {
  mode = newMode;
  document.getElementById("drawBtn").classList.toggle("active", mode === "draw");
  document.getElementById("eraseBtn").classList.toggle("active", mode === "erase");
}

function clearMarkers() {
  for (const i of Object.keys(markerAt)) pixels[Number(i)] = 0;
}

function loadRandomSample() {
  const samples = MODEL_DATA.samples;
  const s = samples[Math.floor(Math.random() * samples.length)];
  pixels = s.pixels.slice();
  sampleInfoEl.textContent = `Loaded a test image of a ${s.label}. Now try adding a fake marker for a different digit.`;
  // suggest a wrong digit for the marker
  markerSelect.value = String((s.label + 1 + Math.floor(Math.random() * 9)) % 10);
  update();
}

function setupControls() {
  for (let d = 0; d < 10; d++) {
    const opt = document.createElement("option");
    opt.value = String(d);
    opt.textContent = String(d);
    markerSelect.appendChild(opt);
  }

  document.getElementById("drawBtn").addEventListener("click", () => setMode("draw"));
  document.getElementById("eraseBtn").addEventListener("click", () => setMode("erase"));
  document.getElementById("clearBtn").addEventListener("click", () => {
    pixels = new Array(N_PIXELS).fill(0);
    sampleInfoEl.textContent = "";
    update();
  });
  document.getElementById("randomBtn").addEventListener("click", loadRandomSample);
  document.getElementById("markerBtn").addEventListener("click", () => {
    clearMarkers();   // one marker at a time keeps it easy to read
    const [r, c] = MODEL_DATA.markerSpots[Number(markerSelect.value)];
    pixels[r * SIZE + c] = 1;
    update();
  });
  document.getElementById("removeMarkersBtn").addEventListener("click", () => {
    clearMarkers();
    update();
  });
}

// ---------------------------------------------------------------- start

buildGrid();
buildModelCards();
setupControls();
loadRandomSample();
