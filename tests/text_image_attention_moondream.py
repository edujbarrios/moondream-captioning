import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from moondream.torch.vision import (
    prepare_crops,
    vision_encoder,
    build_vision_model,
    create_patches,
)
from moondream.torch.text import (
    build_text_model,
    attn as attn_fn,
)
from moondream.torch.config import VisionConfig, TextConfig

# Configuración
st.set_page_config(layout="wide")
st.title("Atención texto → crops (Moondream)")

vision_config = VisionConfig()
text_config = TextConfig()
dtype = torch.float32

vision_model = build_vision_model(vision_config, dtype=dtype)
text_model = build_text_model(text_config, dtype=dtype)

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
caption = st.text_input("Escribe un caption", value="Una mujer caminando por la playa al atardecer.")
run_button = st.button("Ejecutar atención")

if uploaded_file and caption.strip() and run_button:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", width=300)

    crops_tensor, tiling = prepare_crops(image, vision_config, device="cpu")
    crops_tensor = crops_tensor.to(dtype=torch.float32)  # Solución error bfloat16

    st.subheader("Crops y tokens asociados")
    crops_np = crops_tensor.mul(0.5).add(0.5).clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy()

    # Embeddings visuales individuales por crop
    all_embeddings = []
    for i in range(crops_tensor.shape[0]):
        crop = crops_tensor[i].unsqueeze(0)  # [1, 3, H, W]
        with torch.no_grad():
            patches = create_patches(crop, vision_config.enc_patch_size)
            emb = vision_model["patch_emb"](patches) + vision_model["pos_emb"]
            emb = emb.mean(dim=1, keepdim=True)  # [1, 1, D]
            all_embeddings.append(emb)

    # Concatenar crops embeddings [1, num_crops, D]
    visual_embeds = torch.cat(all_embeddings, dim=1)  # [1, N, D]

    # Tokenización simple
    tokens = caption.strip().split()
    vocab = {word: i + 100 for i, word in enumerate(set(tokens))}
    input_ids = torch.tensor([[vocab.get(w, 0) for w in tokens]], dtype=torch.long)

    # Embeddings de texto
    text_embeds = torch.nn.functional.embedding(input_ids, text_model.wte)  # [1, T, D]

    # Concatenar visual + texto
    full_input = torch.cat([visual_embeds, text_embeds], dim=1)  # [1, N+T, D]
    position_ids = torch.arange(full_input.shape[1], dtype=torch.long).unsqueeze(0)

    seq_len = full_input.shape[1]
    attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    # Ejecutar atención
    block = text_model["blocks"][0]
    l_attn, attn_weights = attn_fn(
        x=full_input,
        w=block.attn,
        freqs_cis=text_model.freqs_cis,
        kv_cache=None,
        attn_mask=attn_mask,
        n_heads=text_config.n_heads,
        n_kv_heads=text_config.n_kv_heads,
        position_ids=position_ids,
        lora=None,
        return_attn=True,
    )

    attn = attn_weights.mean(dim=1)[0]  # [T+N, T+N]
    text_start = visual_embeds.shape[1]
    crop_tokens_attn = attn[text_start:, :text_start]  # [T, N]

    # Mostrar cada crop con tokens relevantes
    num_crops = crops_tensor.shape[0]
    topk = 2  # Mostrar los top-k tokens por crop

    cols = st.columns(min(num_crops, 4))
    for i in range(num_crops):
        crop_img = crops_np[i]
        # Obtener los top-k tokens que más atención prestan a este crop
        scores = crop_tokens_attn[:, i]
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        tokens_for_crop = [tokens[j] for j in top_indices]

        with cols[i % len(cols)]:
            st.image(crop_img, caption=f"Crop {i}", width=150)
            st.markdown("**Tokens:** " + ", ".join(tokens_for_crop))

    st.subheader("Heatmap global token → crop")

    fig, ax = plt.subplots(figsize=(num_crops, len(tokens) * 0.4))
    sns.heatmap(
        crop_tokens_attn.numpy(),
        xticklabels=[f"Crop {i}" for i in range(num_crops)],
        yticklabels=tokens,
        cmap="viridis",
        annot=False,
        ax=ax,
    )
    st.pyplot(fig)

    st.caption("Visualización de atención de cada token hacia cada crop.")
