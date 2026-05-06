const API = "http://127.0.0.1:8000";

// --- State ---
let datasets        = {};
let selectedCategory = null;
let selectedDataset  = null;
let currentModelType = null;
let learningRate     = 0.01;
let nIterations      = 1000;
let testSize         = 0.2;
let lastTheta        = null;

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  const res = await fetch(`${API}/datasets`);
  datasets = await res.json();
  renderCategoryPicker();
  setupDiscreteSliders();
  document.getElementById("train-btn").addEventListener("click", trainModel);
});

// ─────────────────────────────────────────────
// STEP 01 — CATEGORY PICKER
// ─────────────────────────────────────────────
function renderCategoryPicker() {
  const meta = {
    regression: {
      label:       "Regression",
      description: "Predict a continuous value. Linear Regression via Gradient Descent.",
      tag:         "LINEAR REGRESSION"
    },
    classification: {
      label:       "Classification",
      description: "Predict a class. Logistic Regression with decision boundary.",
      tag:         "LOGISTIC REGRESSION"
    }
  };

  const container = document.getElementById("category-cards");
  container.innerHTML = "";

  for (const [key, info] of Object.entries(meta)) {
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.innerHTML = `
      <h3>${info.label}</h3>
      <p>${info.description}</p>
      <span class="tag">${info.tag}</span>
    `;
    card.addEventListener("click", () => selectCategory(key, card));
    container.appendChild(card);
  }
}

function selectCategory(key, cardEl) {
  selectedCategory = key;
  selectedDataset  = null;
  lastTheta        = null;
  currentModelType = null;

  document.querySelectorAll("#category-cards .dataset-card")
    .forEach(c => c.classList.remove("selected"));
  cardEl.classList.add("selected");

  document.getElementById("step-dataset").classList.remove("hidden");
  document.getElementById("step-train").classList.add("hidden");

  renderDatasetCards(key);
}

// ─────────────────────────────────────────────
// STEP 02 — DATASET CARDS
// ─────────────────────────────────────────────
function renderDatasetCards(category) {
  const tags = {
    diabetes:            "LINEAR · REAL WORLD",
    synthetic_linear:    "LINEAR · SYNTHETIC",
    nonlinear:           "QUADRATIC · SYNTHETIC",
    make_classification: "BINARY · SYNTHETIC",
    make_moons:          "NON-LINEAR · SYNTHETIC",
    make_circles:        "NON-LINEAR · SYNTHETIC"
  };

  const container = document.getElementById("dataset-cards");
  container.innerHTML = "";

  for (const [key, data] of Object.entries(datasets[category])) {
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.innerHTML = `
      <h3>${data.name}</h3>
      <p>${data.description}</p>
      <span class="tag">${tags[key] ?? key}</span>
    `;
    card.addEventListener("click", () => selectDataset(key, data, card));
    container.appendChild(card);
  }
}

function selectDataset(key, data, cardEl) {
  selectedDataset  = key;
  currentModelType = data.model_type;
  lastTheta        = null;

  document.querySelectorAll("#dataset-cards .dataset-card")
    .forEach(c => c.classList.remove("selected"));
  cardEl.classList.add("selected");

  // Reset all result elements
  document.getElementById("step-train").classList.remove("hidden");
  document.getElementById("diverge-warning").classList.add("hidden");
  document.getElementById("equation-bar").classList.add("hidden");
  document.getElementById("accuracy-bar").classList.add("hidden");
  document.getElementById("plot-cost").classList.add("hidden");
  document.getElementById("stats-bar").classList.add("hidden");
  document.getElementById("confusion-matrix").classList.add("hidden");

  plotScatter(data);
  refreshCodePanel();
}

// ─────────────────────────────────────────────
// SCATTER PLOT (pre-training preview)
// ─────────────────────────────────────────────
function plotScatter(data) {
  const is2D = Array.isArray(data.X[0]);
  let traces;

  if (is2D) {
    const class0 = data.X.filter((_, i) => data.y[i] === 0);
    const class1 = data.X.filter((_, i) => data.y[i] === 1);
    traces = [
      {
        x: class0.map(p => p[0]), y: class0.map(p => p[1]),
        mode: "markers", type: "scatter",
        marker: { color: "#5cffe4", size: 5, opacity: 0.7 },
        name: "Class 0"
      },
      {
        x: class1.map(p => p[0]), y: class1.map(p => p[1]),
        mode: "markers", type: "scatter",
        marker: { color: "#ff6b6b", size: 5, opacity: 0.7 },
        name: "Class 1"
      }
    ];
  } else {
    traces = [{
      x: data.X, y: data.y,
      mode: "markers", type: "scatter",
      marker: { color: "#5cffe4", size: 5, opacity: 0.6 },
      name: "Data points"
    }];
  }

  Plotly.newPlot("plot-scatter", traces, getLayout("Dataset Preview"), { responsive: true });
}

// ─────────────────────────────────────────────
// DISCRETE SLIDERS
// ─────────────────────────────────────────────
function setupDiscreteSliders() {
  setupSliderGroup("lr-slider", "lr-val", val => {
    learningRate = parseFloat(val);
    refreshCodePanel();
  });
  setupSliderGroup("iter-slider", "iter-val", val => {
    nIterations = parseInt(val);
    refreshCodePanel();
  });
  setupSliderGroup("split-slider", "split-val", val => {
    testSize = parseFloat(val);
    const train = Math.round((1 - testSize) * 100);
    const test  = Math.round(testSize * 100);
    document.getElementById("split-val").textContent = `${train} / ${test}`;
    refreshCodePanel();
  });
}

function setupSliderGroup(groupId, labelId, onChange) {
  const group = document.getElementById(groupId);
  group.querySelectorAll(".ds-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".ds-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const val = btn.dataset.val;
      if (groupId !== "split-slider") {
        document.getElementById(labelId).textContent = val;
      }
      onChange(val);
    });
  });
}

// ─────────────────────────────────────────────
// TRAIN
// ─────────────────────────────────────────────
async function trainModel() {
  if (!selectedDataset) return;

  const btn = document.getElementById("train-btn");
  btn.textContent = "Training...";
  btn.disabled = true;

  document.getElementById("diverge-warning").classList.add("hidden");
  document.getElementById("equation-bar").classList.add("hidden");
  document.getElementById("accuracy-bar").classList.add("hidden");
  document.getElementById("stats-bar").classList.add("hidden");
  document.getElementById("confusion-matrix").classList.add("hidden");
  document.getElementById("plot-cost").classList.add("hidden");

  try {
    const res = await fetch(`${API}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_name:  selectedDataset,
        learning_rate: learningRate,
        n_iterations:  nIterations,
        test_size:     testSize
      })
    });

    const result = await res.json();
    lastTheta = result.diverged ? null : result.theta;

    if (result.diverged) {
      document.getElementById("diverge-warning").classList.remove("hidden");
      plotDivergenceMessage();
    } else {
      if (result.model_type === "linear") {
        plotLinearResult(result);
        updateEquation(result.theta);
      } else {
        plotDecisionBoundary(result);
        updateMetrics(result);
      }
      updateStatsBar(result.cost_history);
      plotCostCurve(result.cost_history);
    }

    refreshCodePanel();

  } catch (err) {
    console.error("Training failed:", err);
  } finally {
    btn.textContent = "Train Model";
    btn.disabled = false;
  }
}

// ─────────────────────────────────────────────
// PLOT FUNCTIONS
// ─────────────────────────────────────────────
function plotLinearResult(result) {
  const traces = [
    {
      x: result.train.X, y: result.train.y,
      mode: "markers", type: "scatter",
      marker: { color: "#5cffe4", size: 5, opacity: 0.7 },
      name: "Train"
    },
    {
      x: result.test.X, y: result.test.y,
      mode: "markers", type: "scatter",
      marker: { color: "#ff9f43", size: 5, opacity: 0.7 },
      name: "Test"
    },
    {
      x: result.fit_line.x, y: result.fit_line.y,
      mode: "lines", type: "scatter",
      line: { color: "#ff6b6b", width: 2.5 },
      name: "Fitted line"
    }
  ];
  Plotly.newPlot("plot-scatter", traces, getLayout("Model Fit"), { responsive: true });
}

function plotDecisionBoundary(result) {
  const contour = {
    type: "contour",
    x: result.boundary_grid.xx[0],
    y: result.boundary_grid.yy.map(row => row[0]),
    z: result.boundary_grid.zz,
    colorscale: [[0, "#0a1f2e"], [0.5, "#13161b"], [1, "#2e0a0a"]],
    contours: { start: 0.5, end: 0.5, coloring: "fill" },
    showscale: false,
    opacity: 0.5,
    name: "Decision region"
  };

  const tX = result.train.X;
  const tY = result.train.y;
  const eX = result.test.X;
  const eY = result.test.y;

  const train0 = tX.filter((_, i) => tY[i] === 0);
  const train1 = tX.filter((_, i) => tY[i] === 1);
  const test0  = eX.filter((_, i) => eY[i] === 0);
  const test1  = eX.filter((_, i) => eY[i] === 1);

  const traces = [
    contour,
    {
      x: train0.map(p => p[0]), y: train0.map(p => p[1]),
      mode: "markers", type: "scatter",
      marker: { color: "#5cffe4", size: 6, opacity: 0.8 },
      name: "Train Class 0"
    },
    {
      x: train1.map(p => p[0]), y: train1.map(p => p[1]),
      mode: "markers", type: "scatter",
      marker: { color: "#ff6b6b", size: 6, opacity: 0.8 },
      name: "Train Class 1"
    },
    {
      x: test0.map(p => p[0]), y: test0.map(p => p[1]),
      mode: "markers", type: "scatter",
      marker: { color: "#5cffe4", size: 9, opacity: 1, symbol: "diamond" },
      name: "Test Class 0"
    },
    {
      x: test1.map(p => p[0]), y: test1.map(p => p[1]),
      mode: "markers", type: "scatter",
      marker: { color: "#ff6b6b", size: 9, opacity: 1, symbol: "diamond" },
      name: "Test Class 1"
    }
  ];

  Plotly.newPlot("plot-scatter", traces, getLayout("Decision Boundary"), { responsive: true });
}

function plotDivergenceMessage() {
  const layout = {
    ...getLayout("Cost Function over Iterations"),
    annotations: [{
      text: "⚠ GRADIENT DESCENT DIVERGED<br><span style='font-size:11px'>Learning rate too high — cost exploded to NaN</span>",
      xref: "paper", yref: "paper",
      x: 0.5, y: 0.5,
      showarrow: false,
      font: { color: "#ff6b6b", size: 16, family: "IBM Plex Mono" },
      align: "center"
    }]
  };
  document.getElementById("plot-cost").classList.remove("hidden");
  Plotly.newPlot("plot-cost", [], layout, { responsive: true });
}

function plotCostCurve(costHistory) {
  const trace = {
    y: costHistory,
    x: costHistory.map((_, i) => i + 1),
    mode: "lines", type: "scatter",
    line: { color: "#5cffe4", width: 2 },
    name: "Cost"
  };
  document.getElementById("plot-cost").classList.remove("hidden");
  Plotly.newPlot("plot-cost", [trace], getLayout("Cost Function over Iterations"), { responsive: true });
}

// ─────────────────────────────────────────────
// RESULT DISPLAYS
// ─────────────────────────────────────────────
function updateEquation(theta) {
  const coef      = Math.abs(theta[1]).toFixed(3);
  const intercept = Math.abs(theta[0]).toFixed(3);
  const sign      = theta[1] >= 0 ? "+" : "−";
  document.getElementById("equation-text").textContent =
    `ŷ = ${coef}x ${sign} ${intercept}`;
  document.getElementById("equation-bar").classList.remove("hidden");
}

function updateMetrics(result) {
  document.getElementById("accuracy-val").textContent =
    (result.accuracy * 100).toFixed(1) + "%";
  document.getElementById("accuracy-bar").classList.remove("hidden");

  const cm = result.confusion_matrix;
  document.getElementById("cm-tn").textContent = cm[0][0];
  document.getElementById("cm-fp").textContent = cm[0][1];
  document.getElementById("cm-fn").textContent = cm[1][0];
  document.getElementById("cm-tp").textContent = cm[1][1];
  document.getElementById("confusion-matrix").classList.remove("hidden");
}

function updateStatsBar(costHistory) {
  const start     = costHistory[0];
  const final     = costHistory[costHistory.length - 1];
  const reduction = (((start - final) / start) * 100).toFixed(1);

  document.getElementById("stat-start").textContent     = start.toFixed(4);
  document.getElementById("stat-final").textContent     = final.toFixed(4);
  document.getElementById("stat-reduction").textContent = `${reduction}%`;
  document.getElementById("stats-bar").classList.remove("hidden");
}

// ─────────────────────────────────────────────
// DYNAMIC CODE PANEL
// ─────────────────────────────────────────────
function generateCode(modelType, lr, iterations, testSize, theta) {
  const trainPct = Math.round((1 - testSize) * 100);
  const testPct  = Math.round(testSize * 100);

  if (modelType === "linear") {
    return [
      {
        title: "Imports & Setup",
        code:
`import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler`
      },
      {
        title: "Data Scaling",
        code:
`scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).flatten()`
      },
      {
        title: `Train / Test Split  (${trainPct} / ${testPct})`,
        code:
`X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=${testSize},
    random_state=42
)`
      },
      {
        title: `Gradient Descent  (lr=${lr}, iterations=${iterations})`,
        code:
`def gradient_descent(X, y, learning_rate=${lr}, n_iterations=${iterations}):
    n_samples = len(y)
    theta     = np.zeros(2)
    X_b       = np.c_[np.ones(n_samples), X]
    cost_history = []
    diverged  = False

    for i in range(${iterations}):
        predictions = X_b.dot(theta)
        errors      = predictions - y
        gradients   = (1 / n_samples) * X_b.T.dot(errors)
        theta       = theta - ${lr} * gradients
        cost        = (1 / (2 * n_samples)) * np.sum(errors ** 2)

        if np.isinf(cost) or np.isnan(cost):
            diverged = True
            break

        cost_history.append(cost)

    return theta, cost_history, diverged

theta, cost_history, diverged = gradient_descent(X_train, y_train)`
      },
      {
        title: "Prediction & Evaluation",
        code: theta
          ?
`X_test_b = np.c_[np.ones(len(X_test)), X_test]
y_pred   = X_test_b.dot(theta)
mse      = np.mean((y_pred - y_test) ** 2)

# Results from your current run:
print(f"Intercept   : ${theta[0].toFixed(4)}")
print(f"Coefficient : ${theta[1].toFixed(4)}")
print(f"MSE         : {mse:.4f}")`
          :
`X_test_b = np.c_[np.ones(len(X_test)), X_test]
y_pred   = X_test_b.dot(theta)
mse      = np.mean((y_pred - y_test) ** 2)

print(f"Intercept   : {theta[0]:.4f}")
print(f"Coefficient : {theta[1]:.4f}")
print(f"MSE         : {mse:.4f}")`
      }
    ];
  }

  // ── LOGISTIC ──
  return [
    {
      title: "Imports & Setup",
      code:
`import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))`
    },
    {
      title: "Data Scaling",
      code:
`scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)`
    },
    {
      title: `Train / Test Split  (${trainPct} / ${testPct})`,
      code:
`X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=${testSize},
    random_state=42
)`
    },
    {
      title: `Gradient Descent  (lr=${lr}, iterations=${iterations})`,
      code:
`def gradient_descent_logistic(X, y, learning_rate=${lr}, n_iterations=${iterations}):
    n_samples  = len(y)
    theta      = np.zeros(X.shape[1] + 1)   # bias + 2 weights
    X_b        = np.c_[np.ones(n_samples), X]
    cost_history = []
    diverged   = False

    for i in range(${iterations}):
        z           = X_b.dot(theta)
        predictions = sigmoid(z)
        errors      = predictions - y
        gradients   = (1 / n_samples) * X_b.T.dot(errors)
        theta       = theta - ${lr} * gradients

        p    = np.clip(predictions, 1e-10, 1 - 1e-10)
        cost = -1/n_samples * np.sum(
                   y * np.log(p) + (1 - y) * np.log(1 - p)
               )

        if np.isinf(cost) or np.isnan(cost):
            diverged = True
            break

        cost_history.append(cost)

    return theta, cost_history, diverged

theta, cost_history, diverged = gradient_descent_logistic(X_train, y_train)`
    },
    {
      title: "Prediction & Evaluation",
      code: theta
        ?
`X_test_b    = np.c_[np.ones(len(X_test)), X_test]
y_pred_prob = sigmoid(X_test_b.dot(theta))
y_pred      = (y_pred_prob >= 0.5).astype(int)
accuracy    = np.mean(y_pred == y_test)

# Results from your current run:
print(f"Accuracy : ${(lastTheta ? "—" : "—")}")
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))
TP = np.sum((y_pred == 1) & (y_test == 1))
print(f"TN={TN}  FP={FP}  /  FN={FN}  TP={TP}")`
        :
`X_test_b    = np.c_[np.ones(len(X_test)), X_test]
y_pred_prob = sigmoid(X_test_b.dot(theta))
y_pred      = (y_pred_prob >= 0.5).astype(int)
accuracy    = np.mean(y_pred == y_test)

print(f"Accuracy : {accuracy:.4f}")
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))
TP = np.sum((y_pred == 1) & (y_test == 1))
print(f"TN={TN}  FP={FP}  /  FN={FN}  TP={TP}")`
    }
  ];
}

function refreshCodePanel() {
  if (!selectedDataset || !currentModelType) return;

  const sections = generateCode(currentModelType, learningRate, nIterations, testSize, lastTheta);
  const panel    = document.getElementById("code-panel");

  const openStates = {};
  panel.querySelectorAll(".code-section").forEach((el, i) => {
    openStates[i] = el.classList.contains("open");
  });

  panel.innerHTML = "";

  sections.forEach((section, i) => {
    const isOpen = openStates[i] ?? (i === 3);
    const div    = document.createElement("div");
    div.className = `code-section ${isOpen ? "open" : ""}`;
    div.innerHTML = `
      <div class="code-section-header">
        <span class="code-section-title">
          <span class="section-num">0${i + 1}</span>${section.title}
        </span>
        <span class="code-toggle">▼</span>
      </div>
      <div class="code-section-body">
        <pre><code class="language-python">${escapeHtml(section.code)}</code></pre>
      </div>
    `;
    div.querySelector(".code-section-header").addEventListener("click", () => {
      div.classList.toggle("open");
    });
    panel.appendChild(div);
  });

  panel.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ─────────────────────────────────────────────
// SHARED PLOTLY LAYOUT
// ─────────────────────────────────────────────
function getLayout(title) {
  return {
    title: { text: title, font: { color: "#5a5f72", size: 13, family: "IBM Plex Mono" } },
    paper_bgcolor: "#13161b",
    plot_bgcolor:  "#13161b",
    font:    { color: "#e8eaf0", family: "IBM Plex Mono" },
    margin:  { t: 40, r: 20, b: 40, l: 50 },
    xaxis:   { gridcolor: "#1f2330", zerolinecolor: "#1f2330" },
    yaxis:   { gridcolor: "#1f2330", zerolinecolor: "#1f2330" },
    legend:  { bgcolor: "transparent", font: { size: 11 } }
  };
}
