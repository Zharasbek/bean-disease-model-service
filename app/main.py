from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


MODEL_ID = "Zharasbek/resnet18-bean-disease-classifier"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once when the API starts.
    """
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading model: {MODEL_ID}")

    app.state.classifier = pipeline(
        task="image-classification",
        model=MODEL_ID,
        device=device,
    )
    app.state.device = "cuda" if device == 0 else "cpu"

    print(f"Model loaded on {app.state.device}")

    yield

    # Runs when the API shuts down.
    del app.state.classifier


app = FastAPI(
    title="Bean Disease Classification API",
    description="Classifies bean-leaf images as angular leaf spot, bean rust, or healthy.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Bean Disease Classification API",
        "documentation": "/docs",
    }


@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": request.app.state.device,
    }


@app.post("/predict")
def predict(
    request: Request,
    file: UploadFile = File(...),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and WebP images are supported.",
        )

    image_data = file.file.read(MAX_FILE_SIZE + 1)

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The image must be 10 MB or smaller.",
        )

    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    predictions = request.app.state.classifier(
        image,
        top_k=3,
    )

    return {
        "filename": file.filename,
        "model": MODEL_ID,
        "predictions": [
            {
                "label": prediction["label"],
                "score": round(float(prediction["score"]), 6),
            }
            for prediction in predictions
        ],
    }
