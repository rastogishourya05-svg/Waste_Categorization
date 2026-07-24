"""
Streamlit demo UI - upload an image, see detected waste items with bounding boxes.

Run: streamlit run demo/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from PIL import Image
from src.inference.predict import detect, annotate_image

st.set_page_config(page_title="Waste Detector", layout="centered")
st.title("♻️ Waste Detection & Categorization")
st.write("Upload an image to detect and classify waste items.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected")
        annotated = annotate_image(image, conf=conf)
        st.image(annotated, use_container_width=True)

    detections = detect(image, conf=conf)
    st.subheader(f"Found {len(detections)} item(s)")
    for d in detections:
        st.write(f"**{d['class']}** — confidence: {d['confidence']:.2f}")
