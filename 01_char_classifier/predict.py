import argparse

import torch
from torchvision import transforms
from PIL import Image, ImageOps

from model import CharCNN

IMG_SIZE = 32

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = CharCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes


def crop_to_content(img: Image.Image, padding_ratio: float = 0.2) -> Image.Image:
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()

    if bbox is None:
        return img

    left, top, right, bottom = bbox
    w, h = right - left, bottom - top

    side = max(w, h)
    pad = int(side * padding_ratio)
    side += pad * 2

    cx, cy = (left + right) // 2, (top + bottom) // 2
    half = side // 2

    crop_box = (cx - half, cy - half, cx + half, cy + half)
    return _crop_with_bounds(img, crop_box)


def _crop_with_bounds(img: Image.Image, box) -> Image.Image:
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    canvas = Image.new("L", (w, h), color=255)
    src_left, src_top = max(left, 0), max(top, 0)
    src_right, src_bottom = min(right, img.width), min(bottom, img.height)
    if src_right > src_left and src_bottom > src_top:
        region = img.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(region, (src_left - left, src_top - top))
    return canvas


def predict_image(model, classes, image_path, device, no_crop: bool = False):
    img = Image.open(image_path).convert("L")

    if not no_crop:
        img = crop_to_content(img)

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = torch.topk(probs, k=5)

    print(f"\nPredicción para {image_path}:")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  '{classes[idx]}'  {prob.item()*100:5.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable auto-cropping to content (use the image as is)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model, classes = load_model(args.checkpoint, device)
    predict_image(model, classes, args.image, device, no_crop=args.no_crop)