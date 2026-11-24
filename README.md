# Hydroponic Crops Disease and Condition Classification Using Deep Learning

A computer-vision project that uses a MobileNetV2-based deep learning model to classify uploaded plant-leaf images into the classes learned during training. The project includes model-training scripts, a FastAPI inference server, and a browser-based dashboard.

> **Note:** Predictions are model outputs and should not be treated as confirmed plant-disease diagnoses.

## Project overview

The system accepts a plant-leaf image through a web dashboard. The FastAPI backend validates the upload, resizes the image, runs inference with the trained Keras model, and returns the predicted plant, condition, confidence score, and top-five class predictions.

### Main components

- **Model:** MobileNetV2 with ImageNet-pretrained weights and a custom classification head.
- **Training:** TensorFlow/Keras scripts for model initialization and continued/recovery training.
- **Backend:** FastAPI endpoints for health checks and image prediction.
- **Frontend:** HTML, CSS, and JavaScript dashboard served by FastAPI.
- **Model assets:** A trained `.keras` model and a JSON file containing the output class names.

## Repository structure

```text
Hydroponic-Crops-Disease-and-Condition-Classification/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── model_training/
│   ├── model_creation.py
│   ├── recovery_training.py
│   └── README.md
└── server/
    ├── main.py
    ├── model/
    │   ├── initial_best.keras
    │   ├── class_names.json
    │   └── README.md
    └── frontend/
        ├── index.html
        ├── style.css
        └── script.js
```

## Model architecture

The supplied model-creation script builds a single-stage multiclass classifier:

1. Input image: `160 × 160 × 3`.
2. Data augmentation: random horizontal flip, rotation, and zoom during training.
3. MobileNetV2 backbone: ImageNet-pretrained weights, with the backbone initially frozen.
4. Global Average Pooling.
5. Dropout with rate `0.3`.
6. Dense output layer with `NUM_CLASSES` units and softmax activation.

The model uses Adam with a learning rate of `0.001` and sparse categorical cross-entropy. MobileNetV2 preprocessing is included inside the model graph. Therefore, the inference server passes resized RGB pixel values in the `0–255` range to the model and does not apply a second normalization step.

The number and order of output classes are read from `server/model/class_names.json`. The class-name order must match the output-neuron order used when the model was trained.

## Training scripts

The `model_training/` directory contains:

- `model_creation.py`: loads the train, validation, and test directory datasets; writes the class-name list; builds and compiles the MobileNetV2 model.
- `recovery_training.py`: loads an existing `initial_best.keras` checkpoint, continues training with callbacks, and saves `final_model.keras`.

**Training-script scope:** The supplied `model_creation.py` ends after model compilation; it does not call `model.fit()` or create the initial checkpoint. The recovery script expects `initial_best.keras` to exist already. The training scripts use a Google Colab/Google Drive project path and may need path adjustments to run in another environment.

The dataset itself is not included in this repository. Prepare your own dataset with `train/`, `validation/`, and `test/` directories, each containing one subdirectory per class, before running the training workflow.

## Backend API

The FastAPI application is in `server/main.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Reports whether the model is loaded and the number of classes |
| POST | `/api/predict` | Accepts an image and returns the prediction and top-five results |
| GET | `/` | Serves the dashboard |

The prediction endpoint accepts JPEG, PNG, WEBP, and BMP images up to 10 MB. The server checks the model and class-label files at startup and verifies that the model output count matches the number of labels.

## Run locally

### 1. Requirements

- Python environment compatible with the installed TensorFlow release.
- The repository files.
- `server/model/initial_best.keras`.
- `server/model/class_names.json`.

### 2. Create and activate a virtual environment

**Windows (PowerShell or Command Prompt):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API and dashboard

Run this command from the repository root:

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Open the dashboard:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 5. Check server status

Open:

```text
http://127.0.0.1:8000/api/health
```

When startup succeeds, the response should report `"status": "ready"` and `"model_loaded": true`.

## Prediction response

A successful `/api/predict` request returns:

- Predicted plant name.
- Predicted condition name.
- Predicted class label.
- Confidence score.
- Optional low-confidence warning.
- Top-five class predictions.

The plant and condition are parsed from class labels using the first underscore as the separator. Labels should therefore follow the expected `Plant_Condition` naming convention.

## Dataset and model assets

The dataset is not distributed in this repository. Refer to the original dataset sources and their licences before downloading or redistributing data.

The trained model and `class_names.json` are runtime assets. Keep the JSON class ordering aligned with the model's output layer. If model weights are changed, update the class-name file from the matching training run.

## Limitations and reproducibility notes

- The supplied training code loads the test dataset but does not calculate test metrics in the shown scripts. Do not interpret training or validation accuracy as test performance.
- No numerical performance claims are made here. Add measured test metrics and a confusion matrix only after running a documented evaluation.
- The recovery script continues from a saved checkpoint; it is not a complete standalone initial-training pipeline.
- Model confidence is not a calibrated probability of a medically or agronomically confirmed diagnosis.
- The inference server currently resizes images directly to `160 × 160`; this may alter aspect ratio.

## Future improvements

- Add a dedicated evaluation script that reports accuracy, precision, recall, F1-score, and a confusion matrix on the held-out test set.
- Add automated tests for API input validation and prediction response structure.
- Make training and inference paths configurable through command-line arguments or environment variables.
- Record dataset provenance, class counts, software versions, and experiment settings.

## License

See the [`LICENSE`](LICENSE) file for the terms that apply to this repository. Dataset and pretrained-weight licences may have separate terms.
