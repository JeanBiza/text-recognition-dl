import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WordDataset, ctc_collate_fn
from model import CRNN
from ctc_decode import greedy_decode, character_error_rate


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, charset, blank_idx, train: bool):
    model.train() if train else model.eval()

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels_concat, label_lengths, texts in loader:
            images = images.to(device)
            labels_concat = labels_concat.to(device)
            batch_size = images.size(0)

            if train:
                optimizer.zero_grad()

            logits = model(images)

            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
            T = log_probs.size(0)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long)

            loss = criterion(log_probs, labels_concat, input_lengths, label_lengths)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = greedy_decode(logits.detach().cpu(), charset, blank_idx)
            all_preds.extend(preds)
            all_targets.extend(texts)

    avg_loss = total_loss / n_batches
    cer = character_error_rate(all_preds, all_targets)
    exact_match = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    return avg_loss, cer, exact_match, list(zip(all_preds[:5], all_targets[:5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="../data/words")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_ds = WordDataset(args.data, split="train")
    val_ds = WordDataset(args.data, split="val")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images | Classes (incl. blank): {train_ds.num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=ctc_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=ctc_collate_fn
    )

    model = CRNN(num_classes=train_ds.num_classes).to(device)
    criterion = nn.CTCLoss(blank=train_ds.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_cer, train_exact, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train_ds.charset, train_ds.blank_idx, train=True
        )
        val_loss, val_cer, val_exact, samples = run_epoch(
            model, val_loader, criterion, optimizer, device, train_ds.charset, train_ds.blank_idx, train=False
        )
        scheduler.step(val_loss)
        dt = time.time() - t0

        marker = ""
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {"model_state": model.state_dict(), "charset": train_ds.charset, "blank_idx": train_ds.blank_idx},
                out_path,
            )
            marker = "  <- best so far, saved"

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.3f} train_CER={train_cer:.3f} | "
            f"val_loss={val_loss:.3f} val_CER={val_cer:.3f} val_exact={val_exact:.3f} | "
            f"{dt:.1f}s{marker}"
        )
        if epoch == 1 or epoch % 5 == 0:
            print("    examples (pred -> target):", samples[:3])

    print(f"\nBest validation CER: {best_cer:.4f}")
    print(f"Checkpoint saved to: {out_path}")


if __name__ == "__main__":
    main()