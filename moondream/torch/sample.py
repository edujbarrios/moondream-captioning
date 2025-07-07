import argparse
import json
import os
import torch

from PIL import Image, ImageDraw
from tqdm import tqdm

from .weights import load_weights_into_model
from .moondream import MoondreamModel, MoondreamConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--prompt", "-p", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--max-tokens", "-t", type=int, default=200)
    parser.add_argument("--sampler", "-s", type=str, default="greedy")
    parser.add_argument("--benchmark", "-b", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Load model.
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        config = MoondreamConfig.from_dict(config)
    else:
        config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)
    model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # Encode image.
    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path)

    if not args.benchmark:
        encoded_image = model.encode_image(image)

        # Query
        print(args.prompt)
        for t in model.query(encoded_image, args.prompt, stream=True)["answer"]:
            print(t, end="", flush=True)
        print()
        print()


