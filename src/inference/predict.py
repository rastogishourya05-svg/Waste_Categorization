"""
Shared inference logic - used by both the FastAPI service and the Streamlit demo,
so detection behavior stays consistent across both.
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = "models/best.pt"   # update after training - copy runs/waste-detect/train/weights/best.pt here
CONF_THRESHOLD = 0.4

_model = None


def load_model(weights_path: str = MODEL_PATH):
    global _model
    if _model is None:
        _model = YOLO(weights_path)
    return _model


def detect(image: Image.Image, conf: float = CONF_THRESHOLD):
    """
    Run detection on a PIL image.
    Returns a list of dicts: [{"class": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...]
    """
    model = load_model()
    results = model.predict(source=np.array(image), conf=conf, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": model.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(x, 2) for x in box.xyxy[0].tolist()],
            })
    return detections


def annotate_image(image: Image.Image, conf: float = CONF_THRESHOLD) -> Image.Image:
    """Returns image with bounding boxes drawn - useful for the Streamlit demo."""
    model = load_model()
    results = model.predict(source=np.array(image), conf=conf, verbose=False)
    annotated = results[0].plot()  # returns BGR numpy array
    return Image.fromarray(annotated[..., ::-1])  # convert to RGB
