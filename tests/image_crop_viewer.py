import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import numpy as np
import torch
import cv2
from math import ceil
from moondream.torch.image_crops import overlap_crop_image, reconstruct_from_crops

st.set_page_config(page_title="Segmentación de imagen", layout="wide")
st.title("Segmentación visual con overlap_crop_image")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Imagen original")
    st.image(image_rgb, width=400)

    col1, col2 = st.columns(2)
    with col1:
        overlap_margin = st.slider("Overlap Margin", 0, 100, 4)
    with col2:
        max_crops = st.slider("Max Crops", 1, 20, 12)

    result = overlap_crop_image(image_rgb, overlap_margin=overlap_margin, max_crops=max_crops)
    crops = result["crops"]
    tiling = result["tiling"]

    st.subheader(f"{len(crops)} crops generados (tiling: {tiling})")

    # Mostrar todos los crops
    st.markdown("### Todos los crops (incluye crop[0])")
    num_cols = 5
    num_rows = ceil(len(crops) / num_cols)

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for i in range(num_cols):
            idx = row * num_cols + i
            if idx < len(crops):
                with cols[i]:
                    st.image(crops[idx], caption=f"Crop {idx}", width=150)

    # Mostrar crops usados por Moondream
    st.markdown("### Crops usados por Moondream (crops[1:])")
    num_cols_md = 5
    num_rows_md = ceil((len(crops) - 1) / num_cols_md)

    for row in range(num_rows_md):
        cols_md = st.columns(num_cols_md)
        for i in range(num_cols_md):
            idx = 1 + row * num_cols_md + i
            if idx < len(crops):
                with cols_md[i]:
                    st.image(crops[idx], caption=f"Moondream Crop {idx}", width=150)

    # Imagen reconstruida
    st.subheader("Imagen reconstruida")
    crops_tensor = [torch.from_numpy(crop) for crop in crops[1:]]
    reconstructed = reconstruct_from_crops(crops_tensor, tiling, overlap_margin=overlap_margin)
    reconstructed_np = reconstructed.numpy().astype(np.uint8)
    st.image(reconstructed_np, caption="Reconstrucción", width=400)
