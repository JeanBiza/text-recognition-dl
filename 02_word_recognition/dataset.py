import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class WordDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.images_dir = self.root / split / "images"

        with open(self.root / "charset.json") as f:
            self.charset = json.load(f)

        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.blank_idx = len(self.charset)
        self.num_classes = len(self.charset) + 1

        with open(self.root / split / "labels.csv", newline="") as f:
            reader = csv.DictReader(f)
            self.samples = [(row["filename"], row["text"]) for row in reader]

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

    def decode_indices(self, indices) -> str:
        return "".join(self.charset[i] for i in indices)

    def __getitem__(self, i):
        filename, text = self.samples[i]
        img = Image.open(self.images_dir / filename).convert("L")
        img = self.transform(img)
        label = self.encode_text(text)
        return img, label, text


def ctc_collate_fn(batch):
    images, labels, texts = zip(*batch)

    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)

    return images, labels_concat, label_lengths, texts


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "../data/words"
    ds = WordDataset(root, split="train")
    print(f"Samples: {len(ds)}  |  Classes (incl. blank): {ds.num_classes}")
    img, label, text = ds[0]
    print(f"Image: {img.shape}  |  Text: '{text}'  |  Label indices: {label.tolist()}")