import argparse
import csv
import random
import string
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from generate_chars import get_available_fonts, CHARSET

IMG_HEIGHT = 32
IMG_WIDTH = 128
MIN_LEN, MAX_LEN = 3, 8


def get_broad_fonts():
    import glob
    import platform

    curated = set(get_available_fonts())

    system = platform.system()
    if system == "Windows":
        search_dirs = [r"C:\Windows\Fonts"]
    elif system == "Darwin":
        search_dirs = ["/System/Library/Fonts", "/Library/Fonts"]
    else:
        search_dirs = ["/usr/share/fonts"]

    extra = []
    for d in search_dirs:
        extra.extend(glob.glob(str(Path(d) / "**" / "*.ttf"), recursive=True))
        extra.extend(glob.glob(str(Path(d) / "**" / "*.ttc"), recursive=True))

    all_fonts = list(curated) + [f for f in extra if f not in curated]

    valid = []
    for f in all_fonts:
        try:
            font = ImageFont.truetype(f, 20)
            bbox = font.getbbox("Aa0")
            if bbox[2] - bbox[0] > 0:
                valid.append(f)
        except Exception:
            continue
        if len(valid) >= 60:
            break

    return valid if valid else list(curated)


def random_word(rng: random.Random) -> str:
    length = rng.randint(MIN_LEN, MAX_LEN)
    if rng.random() < 0.5:
        pool = string.ascii_letters
    else:
        pool = CHARSET
    return "".join(rng.choice(pool) for _ in range(length))


def render_word(text: str, font_path: str, rng: random.Random) -> Image.Image:
    bg = 255 if rng.random() < 0.85 else rng.randint(225, 250)
    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=bg)
    draw = ImageDraw.Draw(img)

    font_size = rng.randint(14, 26)
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    while text_w > IMG_WIDTH - 6 and font_size > 8:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    max_x_offset = max(2, IMG_WIDTH - text_w - 4)
    x = rng.randint(2, max_x_offset)
    y = (IMG_HEIGHT - text_h) / 2 - bbox[1] + rng.randint(-2, 2)

    fg = rng.randint(0, 25)
    draw.text((x, y), text, fill=fg, font=font)

    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))

    return img


def generate_split(out_dir: Path, split: str, n: int, fonts, rng: random.Random):
    images_dir = out_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        text = random_word(rng)
        font_path = rng.choice(fonts)
        img = render_word(text, font_path, rng)
        filename = f"{i}.png"
        img.save(images_dir / filename)
        rows.append((filename, text))

    with open(out_dir / split / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text"])
        writer.writerows(rows)

    print(f"  {split}: {n} created images in {images_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="../data/words")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fonts = get_broad_fonts()
    print(f"Using {len(fonts)} available fonts.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import json

    charset = sorted(set(string.ascii_letters + string.digits))
    with open(out_dir / "charset.json", "w") as f:
        json.dump(charset, f, indent=2)

    generate_split(out_dir, "train", args.n_train, fonts, rng)
    generate_split(out_dir, "val", args.n_val, fonts, rng)

    print(f"\nDataset created in: {out_dir}")
    print(f"Charset: {len(charset)} chars -> {''.join(charset)}")


if __name__ == "__main__":
    main()