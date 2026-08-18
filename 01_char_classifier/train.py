import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CharDataset
from model import CharCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="../data/chars")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_ds = CharDataset(args.data, split="train", augment=True)
    val_ds = CharDataset(args.data, split="val", augment=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images | Classes: {len(train_ds.classes)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = CharCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        dt = time.time() - t0

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": train_ds.classes},
                out_path,
            )
            marker = "  <- best so far, saved"

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{dt:.1f}s{marker}"
        )

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to: {out_path}")

if __name__ == "__main__":
    main()
