# Finetuning Moondream 2B

This readme will walk you through the process of finetuning Moondream for better captioning on a given dataset. 

> Make sure to run all commands from the root directory of the project.


# Check before initial setup
## Fine Tuning settings

The fine tuning settings can be modified at:
```bash
moondream/finetune
```
To Fine Tune a model, you must structure your dataset in the following format:

### As JSON:

```json
[
  {
    "image": "images/dog_in_field.jpg",
    "description": "A brown dog is running through a green field under the bright sun."
  },
  {
    "image": "images/children_playing_soccer.jpg",
    "description": "Two children are playing soccer in the park on a sunny afternoon."
  }
]


### As JSONL: (The best format to use)

```jsonl
{"image": "images/dog_in_field.jpg", "description": "A brown dog is running through a green field under the bright sun."}
{"image": "images/children_playing_soccer.jpg", "description": "Two children are playing soccer in the park on a sunny afternoon."}
```jsonl
{"image": "images/dog_in_field.jpg", "description": "A brown dog is running through a green field under the bright sun."}
{"image": "images/children_playing_soccer.jpg", "description": "Two children are playing soccer in the park on a sunny afternoon."}
```


## Initial Setup

### 1.1 Clone and Setup Environment (Linux)
```bash
git clone https://github.com/edujbarrios/moondream-captioning
cd moondream
python -m venv .venv
source .venv/bin/activate
```

### 1.2 Clone and Setup Environment (Windows)
```bash
git clone https://github.com/edujbarrios/moondream-captioning
cd moondream-captioning
python -m venv .venv
.venv\Scripts\activate
.venv\Scripts\Activate.ps1
```

### Install Dependencies 
```bash
# Install base requirements
pip install -r requirements.txt
```

## Downloading the Base Model

Download `model.safetensors` from the [Hugging Face repository](https://huggingface.co/vikhyatk/moondream2/tree/main) and place it in the `models` directory as `moondream_base.safetensors`.

```bash
# Create models directory
mkdir -p models

# Download it using curl (run from root moondream directory)
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors
```

## Weights & Biases

We use Weights & Biases (wandb) to track finetuning progress.

To set it up to track your runs, use `wandb login`.

This will take you through creating an account if you don't have one setup already. Enter your API key and you're ready to go.

## Finetuning the Text Encoder 

For this example, we will be teaching Moondream to describe images. 

Given the prompt: 
`\n\nQuestion: Describe this image.\n\nAnswer:`

We return a more detailed caption of the image then you would get from the base model.

1. Double check that you've updated MODEL_PATH to point to the base moondream model in `moondream/finetune/finetune_text.py`
2. Double check that the save path ends in `.safetensors`, otherwise the run will fail.


### Start Text Finetuning
```bash
python -m moondream.finetune.example_doc_ft
# python -m moondream.finetune.finetune_script_modified   DON'T USE THIS WAY BY NOW
```

The process will output a finetuned version of Moondream into your save path. Example output: `models/moondream_text_finetuned.safetensors`.

### Test the Finetuned Text Encoder

You can test the finetuned models performance with the following command (run from root moondream directory).

This will return the caption of the image.

```bash
# Remember to update the paths
python -m moondream.torch.sample --model [FINETUNED_MODEL_PATH] --image "[DATASET_DIRECTORY]/test/[IMAGE_NAME]" --prompt "\n\nQuestion: Describe this image.\n\nAnswer:"
```

## Finetuning the Region Encoder

For this example, we will be teaching Moondream to detect railroad cracks in images of a railway. 

Our dataset trains our model such that,

Given the prompt: 
`\n\nDetect: <class_name>\n\n`

We are returned the coordinates of a detected crack in the following format:
```{'objects': [{'x_min': [X_MIN], 'y_min': [Y_MIN], 'x_max': [X_MAX], 'y_max': [Y_MAX]}]}```

### Setup Dataset Dependencies

1. Update MODEL_PATH to point to the base moondream model.
5. Double check that the save path ends in `.safetensors`, otherwise the run will fail.
> Navigate to line 244 in `moondream/finetune/finetune_region.py`.
``` # Add save path
    save_file(
        model.state_dict(),
        "moondream_finetune.safetensors", // update this line
    )
```

### Start Region Finetuning
```bash
python -m moondream.finetune.finetune_region
```

The process will output a finetuned version of Moondream into your save path. Example output: `models/moondream_region_finetuned.safetensors`.