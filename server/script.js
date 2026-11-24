const imageInput = document.getElementById("imageInput");
const dropArea = document.getElementById("dropArea");
const imagePreview = document.getElementById("imagePreview");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");

const predictButton = document.getElementById("predictButton");
const buttonText = document.getElementById("buttonText");

const emptyResult = document.getElementById("emptyResult");
const predictionResult = document.getElementById("predictionResult");

const plantName = document.getElementById("plantName");
const conditionName = document.getElementById("conditionName");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");

const warningMessage = document.getElementById("warningMessage");
const topPredictions = document.getElementById("topPredictions");

const errorMessage = document.getElementById("errorMessage");
const serverStatus = document.getElementById("serverStatus");

let selectedFile = null;


// SERVER STATUS
async function checkServer() {
    const dot = serverStatus.querySelector(".status-dot");

    try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("Server unavailable");

        const data = await response.json();

        if (data.model_loaded) {
            serverStatus.lastChild.textContent = " Model ready";
            dot.className = "status-dot online";
        } else {
            serverStatus.lastChild.textContent = " Model loading";
            dot.className = "status-dot";
        }
    } catch (error) {
        serverStatus.lastChild.textContent = " Server offline";
        dot.className = "status-dot offline";
    }
}

checkServer();


// FILE VALIDATION AND PREVIEW
function selectImage(file) {
    if (!file) return;

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    ];

    if (!allowedTypes.includes(file.type)) {
        showError("Please select a JPG, PNG, WEBP or BMP image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image size must be less than 10 MB.");
        return;
    }

    selectedFile = file;
    errorMessage.classList.add("hidden");

    const reader = new FileReader();
    reader.onload = function(event) {
        imagePreview.src = event.target.result;
        imagePreview.classList.remove("hidden");
        uploadPlaceholder.classList.add("hidden");
        predictButton.disabled = false;
    };
    reader.readAsDataURL(file);

    resetResults();
}

imageInput.addEventListener("change", function() {
    if (this.files.length > 0) selectImage(this.files[0]);
});


// DRAG AND DROP
dropArea.addEventListener("dragover", function(event) {
    event.preventDefault();
    dropArea.classList.add("drag-over");
});

dropArea.addEventListener("dragleave", function() {
    dropArea.classList.remove("drag-over");
});

dropArea.addEventListener("drop", function(event) {
    event.preventDefault();
    dropArea.classList.remove("drag-over");
    const file = event.dataTransfer.files[0];
    if (file) selectImage(file);
});


// ERROR DISPLAY
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


// RESET RESULTS
function resetResults() {
    emptyResult.classList.remove("hidden");
    predictionResult.classList.add("hidden");
    warningMessage.textContent = "";
    topPredictions.innerHTML = "";
}


// PREDICTION REQUEST
predictButton.addEventListener("click", async function() {
    if (!selectedFile) {
        showError("Please upload an image first.");
        return;
    }

    predictButton.disabled = true;
    buttonText.textContent = "Analyzing image...";
    errorMessage.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction failed.");

        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        predictButton.disabled = false;
        buttonText.textContent = "Analyze Plant Image";
    }
});


// DISPLAY RESULTS
function displayResults(data) {
    emptyResult.classList.add("hidden");
    predictionResult.classList.remove("hidden");

    plantName.textContent = formatName(data.plant);
    conditionName.textContent = formatName(data.condition);
    confidenceValue.textContent = data.confidence.toFixed(2) + "%";

    const confidence = Math.max(0, Math.min(100, data.confidence));
    confidenceBar.style.width = confidence + "%";
    warningMessage.textContent = data.warning || "";
    topPredictions.innerHTML = "";

    data.top_predictions.forEach(function(item) {
        const percentage = Math.max(0, Math.min(100, item.confidence));

        const row = document.createElement("div");
        row.className = "prediction-item";

        const label = document.createElement("div");
        label.className = "prediction-label";

        const name = document.createElement("span");
        name.textContent =
            formatName(item.plant) + " — " + formatName(item.condition);

        const value = document.createElement("strong");
        value.textContent = item.confidence.toFixed(2) + "%";

        label.appendChild(name);
        label.appendChild(value);

        const track = document.createElement("div");
        track.className = "mini-track";

        const fill = document.createElement("div");
        fill.className = "mini-fill";
        fill.style.width = percentage + "%";

        track.appendChild(fill);
        row.appendChild(label);
        row.appendChild(track);
        topPredictions.appendChild(row);
    });
}


// FORMAT CLASS NAMES
function formatName(value) {
    return value
        .replace(/[_-]/g, " ")
        .replace(/\b\w/g, character => character.toUpperCase());
}
