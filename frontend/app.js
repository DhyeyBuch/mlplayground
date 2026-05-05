const API = "http://127.0.0.1:8000";

// --- State ---
let datasets = {};
let selectedDataset = null;
let learningRate = 0.01;
let nIterations = 1000;
let testSize = 0.2;
let lastTheta = null; // stored after training for code section 5

// --- On page load ---
window.addEventListener("DOMContentLoaded", async () => {
  const res = await fetch(`${API}/datasets`);
  datasets = await res.json();
  renderDatasetCards();
  setupDiscreteSliders();
  document.getElementById("train-btn").addEventListener("click", trainModel);
});

// --- Render dataset selector cards ---
function renderDatasetCards() {
  const tags = {
    diabetes: "LINEAR · REAL WORLD",
    synthetic_linear: "LINEAR · SYNTHETIC",
    nonlinear: "NON-LINEAR · SYNTHETIC"
  };

  const container = document.getElementById("dataset-cards");
  container.innerHTML = "";

  for (const [key, data] of Object.entries(datasets)) {
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.innerHTML = `
      <h3>${data.name}</h3>
      <p>${data.description}</p>
      <span class="tag">${tags[key] ?? key}</span>
    `;
    card.addEventListener("click", () => selectDataset(key, card));
    container.appendChild(card);
  }
}

// --- Dataset selection ---
function selectDataset(key, cardEl) {
  selectedDataset = key;
  lastTheta = null;

  document.querySelectorAll(".dataset-card").forEach(c => c.classList.remove("selected"));
  cardEl.classList.add("selected");

  document.getElementById("step-train").classList.remove("hidden");
  document.getElementById("diverge-warning").classList.add("hidden");
  document.getElementById("plot-cost").classList.add("hidden");
  document.getElementById("stats-bar").classList.add("hidden");
  document.getElementById("equation-bar").classList.add("hidden");

  plotScatter(key);
  refreshCodePanel(); // show code immediately on dataset select
}

// --- Scatter plot of raw dataset ---
function plotScatter(key) {
  const d = datasets[key];
  const trace = {
    x: d.X, y: d.y,
    mode: "markers", type: "scatter",
    marker: { color: "#5cffe4", size: 5, opacity: 0.6 },
    name: "Data points"
  };
  Plotly.newPlot("plot-scatter", [trace], getLayout("Dataset Preview"), { responsive: true });
}

// --- Discrete slider setup ---
function setupDiscreteSliders() {
  setupSliderGroup("lr-slider", "lr-val", (val) => {
    learningRate = parseFloat(val);
    refreshCodePanel();
  });

  setupSliderGroup("iter-slider", "iter-val", (val) => {
    nIterations = parseInt(val);
    refreshCodePanel();
  });

  setupSliderGroup("split-slider", "split-val", (val) => {
    testSize = parseFloat(val);
    const train = Math.round((1 - testSize) * 100);
    const test = Math.round(testSize * 100);
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

// --- Train model ---
async function trainModel() {
  if (!selectedDataset) return;

  const btn = document.getElementById("train-btn");
  btn.textContent = "Training...";
  btn.disabled = true;

  document.getElementById("diverge-warning").classList.add("hidden");
  document.getElementById("stats-bar").classList.add("hidden");
  document.getElementById("equation-bar").classList.add("hidden");
  document.getElementById("plot-cost").classList.add("hidden");

  try {
    const res = await fetch(`${API}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_name: selectedDataset,
        learning_rate: learningRate,
        n_iterations: nIterations,
        test_size: testSize
      })
    });

    const result = await res.json();
    lastTheta = result.diverged ? null : result.theta;

    if (result.diverged) {
      document.getElementById("diverge-warning").classList.remove("hidden");
      plotScatterWithLine(result, true);
      plotDivergenceMessage(); // NEW: annotated error plot
    } else {
      plotScatterWithLine(result, false);
      updateEquation(result.theta);        // NEW
      updateStatsBar(result.cost_history); // NEW
      plotCostCurve(result.cost_history);
    }

    refreshCodePanel(); // update code with real theta if available

  } catch (err) {
    console.error("Training failed:", err);
  } finally {
    btn.textContent = "Train Model";
    btn.disabled = false;
  }
}

// --- Plot scatter + fitted line ---
function plotScatterWithLine(result, diverged) {
  const trainTrace = {
    x: result.train.X, y: result.train.y,
    mode: "markers", type: "scatter",
    marker: { color: "#5cffe4", size: 5, opacity: 0.7 },
    name: "Train"
  };

  const testTrace = {
    x: result.test.X, y: result.test.y,
    mode: "markers", type: "scatter",
    marker: { color: "#ff9f43", size: 5, opacity: 0.7 },
    name: "Test"
  };

  const traces = [trainTrace, testTrace];

  if (!diverged) {
    traces.push({
      x: result.fit_line.x, y: result.fit_line.y,
      mode: "lines", type: "scatter",
      line: { color: "#ff6b6b", width: 2.5 },
      name: "Fitted line"
    });
  }

  Plotly.newPlot("plot-scatter", traces, getLayout("Model Fit"), { responsive: true });
}

// --- NEW: Divergence annotation plot ---
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

// --- Plot cost curve ---
function plotCostCurve(costHistory) {
  const trace = {
    y: costHistory,
    x: costHistory.map((_, i) => i + 1),
    mode: "lines", type: "scatter",
    line: { color: "#5cffe4", width: 2 },
    name: "Cost (MSE)"
  };

  document.getElementById("plot-cost").classList.remove("hidden");
  Plotly.newPlot("plot-cost", [trace], getLayout("Cost Function over Iterations"), { responsive: true });
}

// --- NEW: Update equation display ---
function updateEquation(theta) {
  const intercept = theta[0].toFixed(3);
  const coef = theta[1].toFixed(3);
  const sign = theta[1] >= 0 ? "+" : "-";
  const absCoef = Math.abs(theta[1]).toFixed(3);

  document.getElementById("equation-text").textContent =
    `ŷ = ${absCoef}x ${sign} ${Math.abs(intercept)}`;
  document.getElementById("equation-bar").classList.remove("hidden");
}

// --- NEW: Update stats bar ---
function updateStatsBar(costHistory) {
  const start = costHistory[0];
  const final = costHistory[costHistory.length - 1];
  const reduction = (((start - final) / start) * 100).toFixed(1);

  document.getElementById("stat-start").textContent = start.toFixed(4);
  document.getElementById("stat-final").textContent = final.toFixed(4);
  document.getElementById("stat-reduction").textContent = `${reduction}%`;
  document.getElementById("stats-bar").classList.remove("hidden");
}

// --- NEW: Generate dynamic code sections ---
function generateCode(lr, iterations, testSize, theta) {
  const trainPct = Math.round((1 - testSize) * 100);
  const testPct = Math.round(testSize * 100);

  const sections = [
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
`# Normalize features so gradient descent behaves
# consistently regardless of dataset scale
scaler = StandardScaler()
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
    theta = np.zeros(2)          # [bias, weight]
    X_b = np.c_[np.ones(n_samples), X]
    cost_history = []
    diverged = False

    for i in range(${iterations}):
        predictions = X_b.dot(theta)
        errors      = predictions - y
        gradients   = (1 / n_samples) * X_b.T.dot(errors)
        theta       = theta - ${lr} * gradients

        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)

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
        ? `X_test_b = np.c_[np.ones(len(X_test)), X_test]
y_pred   = X_test_b.dot(theta)
mse      = np.mean((y_pred - y_test) ** 2)

# Results from your current run:
print(f"Intercept   : ${theta[0].toFixed(4)}")
print(f"Coefficient : ${theta[1].toFixed(4)}")
print(f"MSE         : {mse:.4f}")`
        : `X_test_b = np.c_[np.ones(len(X_test)), X_test]
y_pred   = X_test_b.dot(theta)
mse      = np.mean((y_pred - y_test) ** 2)

print(f"Intercept   : {theta[0]:.4f}")
print(f"Coefficient : {theta[1]:.4f}")
print(f"MSE         : {mse:.4f}")`
    }
  ];

  return sections;
}

// --- NEW: Render code panel ---
function refreshCodePanel() {
  if (!selectedDataset) return;

  const sections = generateCode(learningRate, nIterations, testSize, lastTheta);
  const panel = document.getElementById("code-panel");

  // Preserve open/closed state across refreshes
  const openStates = {};
  panel.querySelectorAll(".code-section").forEach((el, i) => {
    openStates[i] = el.classList.contains("open");
  });

  panel.innerHTML = "";

  sections.forEach((section, i) => {
    const isOpen = openStates[i] ?? (i === 3); // Gradient Descent open by default

    const div = document.createElement("div");
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

  // Apply syntax highlighting to all new code blocks
  panel.querySelectorAll("pre code").forEach(block => {
    hljs.highlightElement(block);
  });
}

// --- Utility: escape HTML for safe code injection ---
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// --- Shared Plotly layout config ---
function getLayout(title) {
  return {
    title: { text: title, font: { color: "#5a5f72", size: 13, family: "IBM Plex Mono" } },
    paper_bgcolor: "#13161b",
    plot_bgcolor: "#13161b",
    font: { color: "#e8eaf0", family: "IBM Plex Mono" },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: "#1f2330", zerolinecolor: "#1f2330" },
    yaxis: { gridcolor: "#1f2330", zerolinecolor: "#1f2330" },
    legend: { bgcolor: "transparent", font: { size: 11 } }
  };
}
