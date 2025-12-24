const socket = io();
let localModel;
let VOCAB = [];
let trainingChart;

const NOISE_SCALE = 0.05;
const SAVE_PATH = "localstorage://privatext-model-v1"; // Browser storage path

// --- CHART SETUP ---
function initChart() {
  const ctx = document.getElementById("trainingChart").getContext("2d");
  trainingChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Training Loss",
          data: [],
          borderColor: "#e53935",
          backgroundColor: "rgba(229, 57, 53, 0.1)",
          borderWidth: 2,
          tension: 0.3,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        y: { beginAtZero: true },
        x: { display: false }, // Hide X axis numbers for cleaner look
      },
    },
  });
}

// --- CORE FUNCTIONS ---

async function loadVocab() {
  try {
    const response = await fetch("/vocab.json");
    VOCAB = await response.json();
    console.log(`Vocabulary loaded: ${VOCAB.length} words.`);
  } catch (e) {
    console.error("Could not load vocab.json.");
  }
}

function tokenize(text) {
  const words = text.toLowerCase().split(/\s+/);
  const vector = new Array(VOCAB.length).fill(0);
  words.forEach((word) => {
    const index = VOCAB.indexOf(word);
    if (index !== -1) vector[index] = 1;
  });
  return vector;
}

async function buildModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 16,
      inputShape: [VOCAB.length],
      activation: "relu",
    })
  );
  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

// --- INITIALIZATION & PERSISTENCE ---

async function init() {
  await loadVocab();
  if (VOCAB.length === 0) return;

  initChart();

  try {
    // Try to load from browser storage first
    localModel = await tf.loadLayersModel(SAVE_PATH);
    document.getElementById("status").innerText =
      "Restored saved model from browser!";

    // We need to re-compile the model after loading
    localModel.compile({
      optimizer: tf.train.adam(0.01),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
    console.log("Model loaded from LocalStorage.");
  } catch (e) {
    // If no model found, create a new one
    console.log("No saved model found. Creating new one...");
    localModel = await buildModel();
    document.getElementById("status").innerText =
      "New model created. Ready to train.";
  }
}

async function saveModel() {
  if (!localModel) return;
  await localModel.save(SAVE_PATH);
  document.getElementById("status").innerText =
    "Model saved to browser storage!";

  // Visual feedback on button
  const btn = document.querySelector(".save-btn");
  const originalText = btn.innerText;
  btn.innerText = "✅ Saved!";
  setTimeout(() => (btn.innerText = originalText), 2000);
}

async function resetModel() {
  if (confirm("Are you sure you want to delete your local AI brain?")) {
    localStorage.removeItem("privatext-model-v1"); // Clear storage
    location.reload(); // Reload page to start fresh
  }
}

// --- TRAINING & PRIVACY ---

function addDifferentialPrivacy(originalWeights) {
  return tf.tidy(() => {
    return originalWeights.map((w) => {
      const shape = w.shape;
      const noise = tf.randomNormal(shape, 0, NOISE_SCALE);
      return w.add(noise).arraySync();
    });
  });
}

async function trainAgent(label) {
  const text = document.getElementById("trainInput").value;
  if (!text) return;

  document.getElementById("status").innerText = "Training locally...";

  // Reset Chart visual for this session
  trainingChart.data.labels = [];
  trainingChart.data.datasets[0].data = [];
  trainingChart.update();

  const vector = tokenize(text);
  const x = tf.tensor2d([vector]);
  const y = tf.tensor2d([[label]]);

  await localModel.fit(x, y, {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        trainingChart.data.labels.push(epoch + 1);
        trainingChart.data.datasets[0].data.push(logs.loss);
        trainingChart.update();
      },
    },
  });

  const rawWeights = localModel.getWeights();
  const noisyWeights = addDifferentialPrivacy(rawWeights);
  socket.emit("submit-weights", { weights: noisyWeights });

  document.getElementById("status").innerText = "Training done. Updates sent.";
  document.getElementById("trainInput").value = "";

  // Optional: Auto-save after training?
  // await saveModel(); // Uncomment if you want auto-save
}

async function runPrediction() {
  const text = document.getElementById("testInput").value;
  const vector = tokenize(text);

  const currentWeights = localModel.getWeights()[0].arraySync();
  let analysisHTML =
    "<div class='analysis-text'><strong>Word Impact:</strong><br>";

  vector.forEach((val, i) => {
    if (val === 1) {
      const importance =
        currentWeights[i].reduce((a, b) => a + b, 0) / currentWeights[i].length;
      const color = importance > 0 ? "#d32f2f" : "#388e3c";
      analysisHTML += `<span style="color:${color}">${
        VOCAB[i]
      }: ${importance.toFixed(3)}</span> `;
    }
  });
  analysisHTML += "</div>";

  const x = tf.tensor2d([vector]);
  const prediction = localModel.predict(x);
  const score = (await prediction.data())[0];

  const label = score > 0.5 ? "SPAM" : "NORMAL";
  const percentage = score > 0.5 ? score * 100 : (1 - score) * 100;

  document.getElementById("result").innerHTML = `
        <strong>Result: ${label} (${percentage.toFixed(
    1
  )}% confidence)</strong><br><br>
        ${analysisHTML}
    `;
}

socket.on("update-global-model", async (data) => {
  const weightsAsTensors = data.weights.map((w) => tf.tensor(w));
  localModel.setWeights(weightsAsTensors);
  document.getElementById("status").innerText = "Global Model Updated!";
});

init();
