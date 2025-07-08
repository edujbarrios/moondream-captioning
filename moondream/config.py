# config.py

# Model configuration
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"
DEVICE = "cuda"  # Options: "cuda", "cpu"

# Generation settings
MAX_NEW_TOKENS = 128
DO_SAMPLE = True
TEMPERATURE = 0.7  # Controls creativity: lower = more focused, higher = more random

# Gradio UI configuration
TITLE = "Moondream 2 - Image Question Answering"
DESCRIPTION = "Upload an image and ask questions about it. The model will respond using natural language."
TEXTBOX_LINES = 5
