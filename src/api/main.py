"""
FastAPI service exposing the waste detection model.

Run: uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from src.inference.predict import detect, load_model

app = FastAPI(title="Waste Detection API", version="0.1.0")


@app.on_event("startup")
def startup():
    load_model()  # warm up so first request isn't slow


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    detections = detect(image)

    return JSONResponse({
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections,
    })
