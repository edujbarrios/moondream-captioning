# 🌔 moondream

> [!NOTE]
> *****All the original Moondream material corresponds to the original authors AND all the credits belong to them.*****
>
> *****This adaptation has only be made for specific research and academic porpuses.*****

Moondream is a **tiny vision language model capabable of "describing" images.**

This fork aims to:

- Adapt moondream scripts to an specific academic and research use case in the **image-captioning** field.

- NOT work straight-forward, it needs some previous setup, that will be detailed soon.


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

