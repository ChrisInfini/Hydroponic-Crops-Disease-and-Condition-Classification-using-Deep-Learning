# Server and Dashboard

This folder contains the FastAPI inference server and the static HTML/CSS/JavaScript dashboard supplied for the Hydroponic Plant Disease and Condition Classification project.

## Structure

```text
server/
├── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── model/
    └── README.md
```

## Model files required

Before starting the server, place these two files in `server/model/`:

- `initial_best.keras`
- `class_names.json`

The model file is not included in this source bundle. Do not commit large model weights to GitHub without checking repository size limits and any relevant distribution restrictions.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn server.main:app --reload
```

Then open `http://127.0.0.1:8000`.

API endpoints:

- `GET /api/health` — checks whether the model is loaded.
- `POST /api/predict` — accepts an image upload and returns the top five predictions.

## Input preprocessing

The server resizes images to 160 × 160 RGB and sends float32 pixel values in the 0–255 range. This matches the supplied model architecture, which applies `mobilenet_v2.preprocess_input` inside the model itself. Do not add a second normalization step unless the model architecture is changed.

## Notes

- The dashboard is served by FastAPI; it is not a separate Gradio or Streamlit application.
- The server expects class labels to contain an underscore separating plant and condition, for example `Plant_Condition`.
- The server currently accepts JPEG, PNG, WEBP, and BMP uploads up to 10 MB.
- Predictions are model outputs and should not be presented as confirmed plant disease diagnoses.
