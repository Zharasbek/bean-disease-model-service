import sys

import torch
from PIL import Image
from transformers import pipeline


if len(sys.argv) != 2:
    raise SystemExit("Usage: python predict.py path_to_image.jpg")

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    task="image-classification",
    model="./model",
    device=device,
)

predictions = classifier(image, top_k=3)

for prediction in predictions:
    print(
        f"{prediction['label']}: "
        f"{prediction['score']:.4f}"
    )
