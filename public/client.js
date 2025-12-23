const socket = io();
let localModel;

// Expanded vocabulary for better context
const VOCAB = [
  "free",
  "winner",
  "urgent",
  "click",
  "money",
  "deal",
  "gift",
  "offer",
  "cash",
  "prize", // Spam indicators
  "hello",
  "how",
  "hey",
  "meeting",
  "thanks",
  "study",
  "book",
  "tomorrow",
  "coffee",
  "coming", // Normal indicators
];

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
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" })); // Output between 0 and 1

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
  });
  return model;
}

async function init() {
  localModel = await buildModel();
  document.getElementById("status").innerText =
    "Model is ready for local training.";
}

async function trainAgent(label) {
  const text = document.getElementById("trainInput").value;
  if (!text) return;

  document.getElementById("status").innerText = "Training locally...";
  const vector = tokenize(text);
  const x = tf.tensor2d([vector]);
  const y = tf.tensor2d([[label]]);

  await localModel.fit(x, y, { epochs: 15 });

  const weights = localModel.getWeights().map((w) => w.arraySync());
  socket.emit("submit-weights", { weights });

  document.getElementById("status").innerText =
    "Local updates sent to server. Syncing...";
  document.getElementById("trainInput").value = "";
}

async function runPrediction() {
  const text = document.getElementById("testInput").value;
  const vector = tokenize(text);

  // Weight Analysis (XAI - Explainable AI)
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
  document.getElementById("status").innerText =
    "Global Model Updated! Collaboration successful.";
});

init();
