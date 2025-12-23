const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static("public"));

let weightPool = [];
const MIN_REPORTS = 2;

// Federated Averaging (FedAvg) Logic
function federatedAverage(weightsArray) {
  const clientCount = weightsArray.length;
  return weightsArray[0].map((_, layerIndex) => {
    const layerData = weightsArray.map((client) => client[layerIndex]);
    return layerData[0].map((_, valIndex) => {
      if (Array.isArray(layerData[0][valIndex])) {
        return layerData[0][valIndex].map((__, innerIndex) => {
          let sum = 0;
          for (let i = 0; i < clientCount; i++) {
            sum += layerData[i][valIndex][innerIndex];
          }
          return sum / clientCount;
        });
      } else {
        let sum = 0;
        for (let i = 0; i < clientCount; i++) {
          sum += layerData[i][valIndex];
        }
        return sum / clientCount;
      }
    });
  });
}

io.on("connection", (socket) => {
  console.log(`New Agent Connected: ${socket.id}`);

  socket.on("submit-weights", (data) => {
    weightPool.push(data.weights);
    console.log(`Knowledge Received: (${weightPool.length}/${MIN_REPORTS})`);

    if (weightPool.length >= MIN_REPORTS) {
      console.log("Aggregating models into a new Global Brain...");
      const newGlobalWeights = federatedAverage(weightPool);
      io.emit("update-global-model", { weights: newGlobalWeights });
      weightPool = []; // Reset the pool for the next round
    }
  });
});

const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Federated Server running at http://localhost:${PORT}`);
});
