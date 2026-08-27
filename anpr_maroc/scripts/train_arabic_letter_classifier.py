"""Train the dedicated Arabic-letter CNN from labelled plate crops.

The input folder must use one folder per label, for example ``raw/ه/*.png``.
Use real plate crops for every allowed letter; synthetic font samples alone are
not a suitable replacement for this dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anpr_maroc.ocr.arabic_letter_model import ArabicLetterCNN, preprocess_letter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class LetterDataset(Dataset):
    def __init__(self, examples: list[tuple[Path, int]], augment: bool):
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.examples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {path}")
        glyph = preprocess_letter(image)
        if self.augment:
            angle = random.uniform(-5.0, 5.0)
            matrix = cv2.getRotationMatrix2D((48, 48), angle, random.uniform(0.92, 1.08))
            glyph = cv2.warpAffine(glyph, matrix, (96, 96), borderValue=0.0)
            glyph = np.clip(glyph + np.random.normal(0.0, 0.025, glyph.shape), 0.0, 1.0)
        return torch.from_numpy(glyph.astype(np.float32)).unsqueeze(0), label


def collect_examples(data_dir: Path) -> tuple[list[str], list[tuple[Path, int]]]:
    labels = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    examples = [
        (path, label_index)
        for label_index, label in enumerate(labels)
        for path in (data_dir / label).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return labels, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Arabic plate-letter CNN.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/arabic_letter_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-small-dataset", action="store_true", help="Only for pipeline smoke tests, never production.")
    args = parser.parse_args()

    labels, examples = collect_examples(args.data)
    counts = {label: sum(index == i for _, index in examples) for i, label in enumerate(labels)}
    if len(labels) < 2:
        raise ValueError("at least two letter classes are required")
    if not args.allow_small_dataset and min(counts.values()) < 50:
        raise ValueError(f"insufficient real samples per class: {counts}; need at least 50")

    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.15))
    validation, training = examples[:split], examples[split:]
    train_loader = DataLoader(LetterDataset(training, augment=True), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(LetterDataset(validation, augment=False), batch_size=args.batch_size)
    model = ArabicLetterCNN(len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss_fn(model(images), targets).backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                correct += int((model(images).argmax(dim=1) == targets).sum())
                total += len(targets)
        print(f"epoch={epoch + 1}/{args.epochs} validation_accuracy={correct / max(total, 1):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": labels, "state_dict": model.state_dict()}, args.output)
    print(f"Saved model to {args.output}; classes={labels}; counts={counts}")


if __name__ == "__main__":
    main()
