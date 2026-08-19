import argparse

import torch
from torchvision import transforms
from PIL import Image, ImageOps

from model import CRNN
from ctc_decode import greedy_decode

IMG_HEIGHT = 32
IMG_WIDTH = 128


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    charset = ckpt["charset"]
    blank_idx = ckpt["blank_idx"]
    num_classes = len(charset) + 1

    model = CRNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, charset, blank_idx


def prepare_image(img: Image.Image) -> Image.Image:
    import numpy as np

    avg_brightness = np.array(img).mean()
    if avg_brightness < 128:
        img = ImageOps.invert(img)

    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    if bbox is not None:
        img = img.crop(bbox)

    w, h = img.size
    target_height = int(IMG_HEIGHT * 0.7)
    scale = target_height / h
    new_w = min(IMG_WIDTH - 8, max(1, int(w * scale)))
    new_h = target_height
    img = img.resize((new_w, new_h))

    canvas = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
    x_offset = (IMG_WIDTH - new_w) // 2
    y_offset = (IMG_HEIGHT - new_h) // 2
    canvas.paste(img, (x_offset, y_offset))
    return canvas


def predict_image(model, charset, blank_idx, image_path, device):
    img = Image.open(image_path).convert("L")
    img = prepare_image(img)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    text = greedy_decode(logits.cpu(), charset, blank_idx)[0]
    print(f"\nPred for {image_path}: '{text}'")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, charset, blank_idx = load_model(args.checkpoint, device)
    predict_image(model, charset, blank_idx, args.image, device)