import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_PATH = MODEL_DIR / "initial_best.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
CONFIDENCE_WARNING = 60.0

model = None
class_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(f"Class names file not found: {CLASS_NAMES_PATH}")

    print("Loading MobileNetV2 model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    # Support either a list or a dictionary of class names.
    if isinstance(class_names, dict):
        try:
            class_names = [
                class_names[str(i)] for i in range(len(class_names))
            ]
        except KeyError as exc:
            raise RuntimeError(
                "class_names.json dictionary must use consecutive "
                "numeric keys starting at 0."
            ) from exc

    if not isinstance(class_names, list):
        raise RuntimeError(
            "class_names.json must contain a list or indexed dictionary."
        )

    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model has {model.output_shape[-1]} outputs, but "
            f"class_names.json has {len(class_names)} labels."
        )

    print("Model loaded successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Number of classes:", len(class_names))

    yield

    model = None
    class_names = None
    print("Server shutting down.")


app = FastAPI(
    title="Hydroponic Plant Disease AI",
    description="Plant identification and condition classification API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ready" if model is not None else "loading",
        "model_loaded": model is not None,
        "number_of_classes": len(class_names) if class_names else 0,
    }


@app.post("/api/predict")
async def predict_plant(file: UploadFile = File(...)):
    if model is None or class_names is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not ready.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, WEBP or BMP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Try another file.",
        )

    resized = image.resize(IMAGE_SIZE)
    img_array = np.array(resized).astype("float32")

    img_array = np.expand_dims(img_array, axis=0)

    try:
        predictions = model.predict(img_array, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

    if len(predictions) != len(class_names):
        raise HTTPException(
            status_code=500,
            detail="Model output does not match class labels.",
        )

    if not np.all(np.isfinite(predictions)):
        raise HTTPException(
            status_code=500,
            detail="Model returned invalid prediction values.",
        )

    top_indices = np.argsort(predictions)[::-1][:5]
    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    if "_" not in predicted_class:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected class label format: {predicted_class}",
        )

    plant, condition = predicted_class.split("_", 1)

    top_predictions = []
    for idx in top_indices:
        idx = int(idx)
        label = class_names[idx]

        if "_" not in label:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected class label format: {label}",
            )

        top_plant, top_condition = label.split("_", 1)
        top_predictions.append(
            {
                "class_name": label,
                "plant": top_plant,
                "condition": top_condition,
                "confidence": round(float(predictions[idx] * 100), 2),
            }
        )

    return {
        "success": True,
        "filename": file.filename,
        "plant": plant,
        "condition": condition,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "warning": (
            "Low-confidence prediction."
            if confidence < CONFIDENCE_WARNING
            else None
        ),
        "top_predictions": top_predictions,
    }


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)
