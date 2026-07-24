# Waste Detection & Categorization (CV)

Real-time trash/waste detection and classification using YOLOv8, served via FastAPI, with a Streamlit demo UI.

## Project Structure

```
waste-detection/
├── data/
│   ├── raw/              # original downloaded datasets (TACO, TrashNet, etc.)
│   ├── processed/        # cleaned/resized images ready for training
│   └── annotations/      # YOLO-format label files (.txt) + data.yaml
├── models/                # trained weights (.pt files) - gitignore these, too large
├── src/
│   ├── training/
│   │   └── train.py       # YOLO fine-tuning script
│   ├── inference/
│   │   └── predict.py     # standalone inference helper (used by API + Streamlit)
│   └── api/
│       └── main.py        # FastAPI app exposing /predict endpoint
├── notebooks/              # exploration, dataset EDA, Colab-friendly training notebook
├── demo/
│   └── app.py              # Streamlit demo UI
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Workflow

1. **Get data**: Download TACO (github.com/pedropro/TACO) or TrashNet, put raw images in `data/raw/`.
2. **Convert to YOLO format**: TACO ships COCO-style JSON — you'll need to convert to YOLO `.txt` labels (one line per box: `class x_center y_center width height`, normalized 0-1). Roboflow can do this conversion for you if you upload the dataset there instead of doing it by hand.
3. **Write `data/annotations/data.yaml`** (see template below).
4. **Train**: `python src/training/train.py`
5. **Run inference API**: `uvicorn src.api.main:app --reload`
6. **Run demo UI**: `streamlit run demo/app.py`

## data.yaml template

```yaml
path: ../data/processed
train: images/train
val: images/val

names:
  0: plastic
  1: paper
  2: metal
  3: glass
  4: organic
  5: other
```

Adjust class names/count based on the taxonomy you pick (material-based vs item-based — see notes below).

## Notes

- Start with `yolov8n.pt` (nano) or `yolov8s.pt` (small) as pretrained base — don't train from scratch, fine-tune.
- If mAP is low early on, it's almost always a data problem (too few examples per class, inconsistent labeling) before it's a model problem.
- Keep a fixed validation set from day one so you can actually track whether changes help.
# Waste_Detection
# Waste_Categorization
