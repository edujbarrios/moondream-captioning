## This script hasn't been tested on all cases yet, it still uses hugging face datasets, but needs to be adapted to use local datasets.

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from safetensors.torch import save_file
import configparser
import os
from tqdm import tqdm
from datasets import load_dataset
import wandb

from bitsandbytes.optim import AdamW8bit

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden, _lm_head, TextConfig

# ==============================
# Carga de configuración
# ==============================
configp = configparser.ConfigParser()
configp.read('config.ini')

MODEL_PATH = configp['GENERAL']['MODEL_PATH']
ANSWER_EOS = configp['GENERAL']['ANSWER_EOS']
EPOCHS = int(configp['GENERAL']['EPOCHS'])
GRAD_ACCUM_STEPS = int(configp['GENERAL']['GRAD_ACCUM_STEPS'])
BATCH_SIZE = int(configp['GENERAL'].get('BATCH_SIZE', 1))
SEED = int(configp['GENERAL'].get('SEED', 42))

WANDB_PROJECT = configp['WANDB']['PROJECT']

DATASET_NAME = configp['DATASET']['NAME']
DATASET_SPLIT = configp['DATASET']['SPLIT']

OPTIM_CLASS = configp['OPTIMIZER']['CLASS']
LR = float(configp['OPTIMIZER']['LR'])
BETAS = tuple(map(float, configp['OPTIMIZER']['BETAS'].split(',')))
EPS = float(configp['OPTIMIZER']['EPS'])
WEIGHT_DECAY = float(configp['OPTIMIZER'].get('WEIGHT_DECAY', 0.0))
AMSGRAD = configp['OPTIMIZER'].getboolean('AMSGRAD', False)

SCHED_TYPE = configp['LR_SCHEDULE']['TYPE']
WARMUP = float(configp['LR_SCHEDULE'].get('WARMUP', 0.1))
MIN_LR = float(configp['LR_SCHEDULE'].get('MIN_LR', 0.1))
MAX_LR = float(configp['LR_SCHEDULE'].get('MAX_LR', 1.0))

SAVE_PATH = configp['OUTPUT']['SAVE_PATH']

# ==============================
# Funciones auxiliares
# ==============================

def get_lr_schedule_fn(configp, base_lr, total_steps):
    schedule_type = configp['LR_SCHEDULE'].get('TYPE', 'constant').lower()

    if schedule_type == 'constant':
        def schedule(step, max_steps=total_steps):
            return base_lr
        return schedule

    elif schedule_type == 'cosine':
        warmup = float(configp['LR_SCHEDULE'].get('WARMUP', 0.1))
        min_lr = float(configp['LR_SCHEDULE'].get('MIN_LR', 0.1)) * base_lr
        max_lr = float(configp['LR_SCHEDULE'].get('MAX_LR', 1.0)) * base_lr
        def schedule(step, max_steps=total_steps):
            x = step / max_steps
            if x < warmup:
                return min_lr + (max_lr - min_lr) * x / warmup
            else:
                return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * (x - warmup) / (1 - warmup))) / 2
        return schedule

    elif schedule_type == 'linear':
        min_lr = float(configp['LR_SCHEDULE'].get('MIN_LR', 0.1)) * base_lr
        max_lr = float(configp['LR_SCHEDULE'].get('MAX_LR', 1.0)) * base_lr
        def schedule(step, max_steps=total_steps):
            progress = step / max_steps
            return max_lr - (max_lr - min_lr) * progress
        return schedule

    else:
        raise ValueError(f"Schedule '{schedule_type}' not implemented.")

def text_loss(inputs_embeds: torch.Tensor, w: nn.Module, labels: torch.Tensor, config: TextConfig):
    _, q_len, _ = inputs_embeds.shape
    hidden_BTC = _produce_hidden(inputs_embeds, w, config)
    lm_logits = _lm_head(hidden_BTC, w)

    loss = None
    if labels is not None:
        _, _, l_len = labels.shape
        shift_index = (q_len - l_len) - 1
        shifted_logits = lm_logits[..., shift_index:-1, :].contiguous()
        shifted_labels = labels.contiguous()
        loss = nn.CrossEntropyLoss()(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )
    return loss

class DocciDataset(Dataset):
    def __init__(self, split=DATASET_SPLIT):
        self.data = load_dataset(DATASET_NAME, trust_remote_code=True)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        description = sample["description"]
        return {
            "image": sample["image"],
            "qa": {
                "question": "\n\nQuestion: Describe this image.\n\nAnswer:",
                "answer": f"{description}{ANSWER_EOS}",
            },
        }

def build_optimizer(model):
    if OPTIM_CLASS == 'AdamW8bit':
        return AdamW8bit(
            [{"params": model.text.parameters()}],
            lr=LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            amsgrad=AMSGRAD,
        )
    else:
        raise ValueError(f"Optimizador {OPTIM_CLASS} no soportado todavía.")

# ==============================
# Entrenamiento principal
# ==============================

def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project=WANDB_PROJECT,
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
            "OPTIMIZER": OPTIM_CLASS,
            "BETAS": BETAS,
            "EPS": EPS,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "AMSGRAD": AMSGRAD,
            "LR_SCHEDULE_TYPE": SCHED_TYPE,
            "WARMUP": WARMUP,
            "MIN_LR": MIN_LR,
            "MAX_LR": MAX_LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)

    optimizer = build_optimizer(model)
    dataset = DocciDataset(DATASET_SPLIT)
    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    lr_schedule = get_lr_schedule_fn(configp, LR, total_steps)
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            i += 1
            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
            bos_emb = text_encoder(
                torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
                model.text,
            )
            question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
            question_emb = text_encoder(
                torch.tensor([[question_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            answer_emb = text_encoder(
                torch.tensor([[answer_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            inputs_embeds = torch.cat(
                [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
            )
            loss = text_loss(
                inputs_embeds=inputs_embeds,
                w=model.text,
                labels=torch.tensor([[answer_tokens]], device=model.device),
                config=config.text,
            )

            loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                pbar.set_postfix({"step": i // GRAD_ACCUM_STEPS, "loss": loss.item()})
                pbar.update(1)
                wandb.log({"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    wandb.finish()
    # ==============================
    # Crear directorio de guardado si no existe
    # ==============================
    save_dir = os.path.dirname(SAVE_PATH)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_file(
        model.state_dict(),
        SAVE_PATH,
    )

if __name__ == "__main__":
    main()
