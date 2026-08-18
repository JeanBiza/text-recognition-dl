import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CharDataset(Dataset):
    def __init__(self, root: str, split: str = "train", augment: bool = False):
        self.root = Path(root)
        self.split_dir = self.root / split

        with open(self.root / "labels.json") as f:
            idx_to_char = json.load(f)

        def folder_name(c):
            return f"{ord(c):03d}_{'upper' if c.isupper() else 'lower' if c.islower() else 'digit'}"

        self.classes = [idx_to_char[str(i)] for i in range(len(idx_to_char))]
        self.folder_to_idx = {folder_name(c): i for i, c in enumerate(self.classes)}

        self.samples = []
        for folder, idx in self.folder_to_idx.items():
            folder_path = self.split_dir / folder
            if not folder_path.exists():
                continue
            for img_path in folder_path.glob("*.png"):
                self.samples.append((img_path, idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images were found in {self.split_dir}. "
                "Did you run generate_chars.py first?"
            )

        aug = []
        if augment:
            aug = [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            ]

        self.transform = transforms.Compose(
            aug
            + [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, label = self.samples[i]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "../data/chars"
    ds = CharDataset(root, split="train")
    print(f"Samples: {len(ds)}  |  Classes: {len(ds.classes)}")
    x, y = ds[0]
    print(f"Sample shape: {x.shape}  |  Label: {y} ({ds.classes[y]})")
