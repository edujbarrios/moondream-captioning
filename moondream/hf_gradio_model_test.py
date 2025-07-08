import gradio as gr
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

# Load config from config.py
MODEL_ID = config.MODEL_ID
REVISION = config.REVISION
DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = config.MAX_NEW_TOKENS
DO_SAMPLE = config.DO_SAMPLE
TEMPERATURE = config.TEMPERATURE
TITLE = config.TITLE
DESCRIPTION = config.DESCRIPTION
TEXTBOX_LINES = config.TEXTBOX_LINES

# Load tokenizer and model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True).to(DEVICE)
model.eval()

# Function to answer one or multiple questions

def generate_response(image: Image.Image, questions: str, mode: str) -> str:
    if image is None or not questions.strip():
        return "Please upload an image and enter your question(s)."

    all_questions = [questions.strip()] if mode == "Single" else questions.strip().split("\n")
    responses = []

    for question in all_questions:
        prompt = f"<image>\n\nQuestion: {question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Answer:")[-1].strip()
        responses.append(f"Q: {question}\nA: {answer}\n")

    return "\n".join(responses)

# Gradio interface with mode selection
interface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Image(label="Upload an image"),
        gr.Textbox(lines=TEXTBOX_LINES, label="Enter your question(s)"),
        gr.Radio(["Single", "Multiple"], value="Single", label="Question Mode")
    ],
    outputs=gr.Textbox(label="Answer(s)"),
    title=TITLE,
    description=DESCRIPTION
)

interface.launch()
