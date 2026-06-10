const fileInput = document.querySelector("#fileInput");
const dropLabel = document.querySelector("#dropLabel");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const predictButton = document.querySelector("#predictButton");
const clearButton = document.querySelector("#clearButton");
const message = document.querySelector("#message");
const apiStatus = document.querySelector("#apiStatus");
const resultTitle = document.querySelector("#resultTitle");
const decisionBadge = document.querySelector("#decisionBadge");
const rawResponse = document.querySelector("#rawResponse");

const fields = {
  anomalyScore: document.querySelector("#anomalyScore"),
  confidence: document.querySelector("#confidence"),
  totalLatency: document.querySelector("#totalLatency"),
  labelValue: document.querySelector("#labelValue"),
  uncertainValue: document.querySelector("#uncertainValue"),
  classifierCalled: document.querySelector("#classifierCalled"),
  adLatency: document.querySelector("#adLatency"),
  classifierLatency: document.querySelector("#classifierLatency"),
  lsTask: document.querySelector("#lsTask"),
};

let selectedFile = null;
let previewUrl = null;

function formatPercent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

function formatMs(value) {
  return typeof value === "number" ? `${value.toFixed(1)} ms` : "--";
}

function formatBool(value) {
  return typeof value === "boolean" ? (value ? "Yes" : "No") : "--";
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.style.color = isError ? "#b42318" : "";
}

function setApiStatus(text, className) {
  apiStatus.textContent = text;
  apiStatus.className = `status-pill ${className || ""}`.trim();
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    setApiStatus(health.pipeline_loaded ? "API ready" : "Pipeline loading", health.pipeline_loaded ? "ok" : "");
  } catch (error) {
    setApiStatus("API offline", "error");
  }
}

function selectFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setMessage("Choose an image file.", true);
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.style.display = "block";
  emptyPreview.style.display = "none";
  predictButton.disabled = false;
  clearButton.disabled = false;
  setMessage(`${file.name} selected`);
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  emptyPreview.style.display = "block";
  predictButton.disabled = true;
  clearButton.disabled = true;
  setMessage("");
}

function renderResult(result) {
  const isUncertain = result.uncertain === true;
  const isAnomaly = result.anomaly === true;

  resultTitle.textContent = isUncertain ? "Needs review" : isAnomaly ? "Anomaly detected" : "Normal sample";
  decisionBadge.textContent = isUncertain ? "Uncertain" : isAnomaly ? "Anomaly" : "Normal";
  decisionBadge.className = `decision-badge ${isUncertain ? "warn" : isAnomaly ? "bad" : "good"}`;

  fields.anomalyScore.textContent = formatPercent(result.anomaly_score);
  fields.confidence.textContent = formatPercent(result.confidence);
  fields.totalLatency.textContent = formatMs(result.total_latency_ms);
  fields.labelValue.textContent = result.label || (isAnomaly ? "Anomaly" : "Normal");
  fields.uncertainValue.textContent = formatBool(result.uncertain);
  fields.classifierCalled.textContent = formatBool(result.model_b_called);
  fields.adLatency.textContent = formatMs(result.ad_latency_ms);
  fields.classifierLatency.textContent = formatMs(result.classifier_latency_ms);
  fields.lsTask.textContent = result.ls_task_id ?? "--";
  rawResponse.textContent = JSON.stringify(result, null, 2);
}

async function runPrediction() {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled = true;
  setMessage("Running prediction...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed");
    }
    renderResult(payload);
    setMessage("Prediction complete.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    predictButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
predictButton.addEventListener("click", runPrediction);
clearButton.addEventListener("click", clearSelection);

["dragenter", "dragover"].forEach((eventName) => {
  dropLabel.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropLabel.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropLabel.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropLabel.classList.remove("dragging");
  });
});

dropLabel.addEventListener("drop", (event) => {
  selectFile(event.dataTransfer.files[0]);
});

checkHealth();
