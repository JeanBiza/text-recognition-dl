import argparse
import glob
import json
import platform
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

CHARSET = string.digits + string.ascii_uppercase + string.ascii_lowercase

IMG_SIZE = 32

FONT_CANDIDATES_LINUX = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]

FONT_CANDIDATES_WINDOWS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\cambria.ttc",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]

FONT_CANDIDATES_MACOS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def get_available_fonts():
    system = platform.system()

    if system == "Windows":
        candidates = FONT_CANDIDATES_WINDOWS
    elif system == "Darwin":
        candidates = FONT_CANDIDATES_MACOS
    else:
        candidates = FONT_CANDIDATES_LINUX

    fonts = [f for f in candidates if Path(f).exists()]

    if not fonts:
        if system == "Windows":
            search_dirs = [r"C:\Windows\Fonts"]
        elif system == "Darwin":
            search_dirs = ["/System/Library/Fonts", "/Library/Fonts"]
        else:
            search_dirs = ["/usr/share/fonts"]

        for d in search_dirs:
            fonts.extend(glob.glob(str(Path(d) / "**" / "*.ttf"), recursive=True))
            fonts.extend(glob.glob(str(Path(d) / "**" / "*.ttc"), recursive=True))

        fonts = fonts[:40]

    if not fonts:
        raise RuntimeError(
            f"No matching font was found on the system ({system}). "
            "Please install a TrueType font or update FONT_CANDIDATES_* with "
            "valid paths for your machine."
        )
    return fonts


def label_folder_name(char: str) -> str:
    return f"{ord(char):03d}_{'upper' if char.isupper() else 'lower' if char.islower() else 'digit'}"


def render_char(char: str, font_path: str) -> Image.Image:
    canvas_size = IMG_SIZE * 2
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(int(IMG_SIZE * 0.9), int(IMG_SIZE * 1.5))
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((canvas_size - w) / 2 - bbox[0], (canvas_size - h) / 2 - bbox[1])
    draw.text(pos, char, fill=0, font=font)

    angle = random.uniform(-8, 8)
    img = img.rotate(angle, fillcolor=255, resample=Image.BICUBIC)

    left = (canvas_size - IMG_SIZE) // 2
    img = img.crop((left, left, left + IMG_SIZE, left + IMG_SIZE))

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    return img


def generate_dataset(out_dir: Path, per_class: int, val_split: float, seed: int):
    random.seed(seed)
    fonts = get_available_fonts()
    print(f"Using {len(fonts)} available fonts.")

    labels = {str(i): c for i, c in enumerate(CHARSET)}
    (out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    n_val = max(1, int(per_class * val_split))
    n_train = per_class - n_val

    for char in CHARSET:
        folder = label_folder_name(char)
        train_dir = out_dir / "train" / folder
        val_dir = out_dir / "val" / folder
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for split_dir, n in [(train_dir, n_train), (val_dir, n_val)]:
            for i in range(n):
                font_path = random.choice(fonts)
                img = render_char(char, font_path)
                img.save(split_dir / f"{i}.png")

        print(f"  '{char}' -> {n_train} train / {n_val} val")

    print(f"\nDataset created in: {out_dir}")
    print(f"Total of classes: {len(CHARSET)}")
    print(f"Total of images: {len(CHARSET) * per_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="../data/chars", help="Output directory")
    parser.add_argument("--per_class", type=int, default=500, help="Images per char (train+val)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validate fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(Path(args.out), args.per_class, args.val_split, args.seed)
