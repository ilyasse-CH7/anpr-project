"""Fine-tune the plate-letter CNN on real crops, starting from the AHCD model.

Two stages, because the two corpora carry different information:

  1. ``models/arabic_letter_classifier_ahcd.pt`` was trained on 16 000
     handwritten AHCD glyphs.  It never saw a printed plate, but it learned
     what Arabic stroke structure looks like -- and 16 000 samples of that is
     something 165 plate crops cannot teach.
  2. This script replaces its 28-way head and fine-tunes on the real crops from
     ``splits.json``, which is what closes the handwritten/printed domain gap.

The backbone trains at a much lower rate than the new head: with 165 training
images, letting the convolutions move at head speed just memorises them.

Augmentation is applied to the raw colour crop *before* ``preprocess_letter``,
not to the binarised glyph afterwards.  Brightness, contrast and blur are
exactly the capture variations the binarisation has to survive, so applying
them after it would train against noise the real pipeline never produces.
Mirroring is never applied: a mirrored Arabic glyph is a different shape.

    python -m anpr_maroc.scripts.train_letter_classifier_v2
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anpr_maroc.ocr.arabic_letter_model import IMAGE_SIZE, ArabicLetterCNN, preprocess_letter

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = Path("data/arabic_letters/splits.json")
DEFAULT_BASE = Path("models/arabic_letter_classifier_ahcd.pt")
DEFAULT_OUTPUT = Path("models/arabic_letter_classifier_real.pt")


def fallback_glyph(image: np.ndarray) -> np.ndarray:
    """Plain resize for crops where no isolated glyph survives binarisation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def augment_raw(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light, glyph-preserving capture-side jitter."""
    out = image.astype(np.float32)

    # Exposure and contrast: the dominant variation between plate photographs.
    out = out * rng.uniform(0.75, 1.30) + rng.uniform(-28.0, 28.0)
    out = np.clip(out, 0, 255).astype(np.uint8)

    height, width = out.shape[:2]
    angle = rng.uniform(-7.0, 7.0)
    scale = rng.uniform(0.92, 1.08)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-0.03, 0.03) * width
    matrix[1, 2] += rng.uniform(-0.03, 0.03) * height
    out = cv2.warpAffine(out, matrix, (width, height), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.30:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.30:
        out = np.clip(out.astype(np.float32) +
                      np.random.normal(0, rng.uniform(3, 10), out.shape), 0, 255).astype(np.uint8)
    return out


class RealLetterDataset(Dataset):
    def __init__(self, records, classes: list[str], augment: bool, seed: int = 0):
        self.records = records
        self.index = {label: i for i, label in enumerate(classes)}
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        record = self.records[i]
        image = cv2.imread(str(PROJECT / record["path"]))
        if image is None:
            raise RuntimeError(f"unreadable: {record['path']}")
        if self.augment:
            rng = random.Random((self.seed, i, torch.randint(0, 1 << 30, (1,)).item()).__hash__())
            candidate = augment_raw(image, rng)
        else:
            candidate = image
        try:
            glyph = preprocess_letter(candidate)
        except Exception:
            # Augmentation can push a thin glyph against the border, where the
            # frame-removal step discards it.  Fall back to the clean crop
            # rather than dropping the sample from the epoch.
            try:
                glyph = preprocess_letter(image)
            except Exception:
                glyph = fallback_glyph(image)
        return torch.from_numpy(glyph).unsqueeze(0), self.index[record["label"]]


def load_split(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_split = collections.defaultdict(list)
    for record in payload["records"]:
        by_split[record["split"]].append(record)
    return payload["classes"], by_split


# AHCD has no hamza-alif; on a plate the bare alif and the hamza-alif are the
# same field, so its ا supervises our أ.
AHCD_EQUIVALENT = {"أ": "ا"}


class AhcdDataset(Dataset):
    """AHCD glyphs mapped onto the target label set.

    Reusing only the backbone from ``ahcd.pt`` leaves the 4-way head starting
    from noise on 165 images.  Pre-training the *whole* network on the target
    classes gives the head a real starting point: the letters we care about are
    supervised by their AHCD counterparts, and the 25 alphabet letters we do not
    predict become a stand-in for "autre" -- imperfect, since the real "autre"
    is mostly digits, but it teaches the head that not everything is a letter.
    """

    def __init__(self, raw_dir: Path, classes: list[str], per_other: int, seed: int):
        self.index = {label: i for i, label in enumerate(classes)}
        wanted = {AHCD_EQUIVALENT.get(c, c): c for c in classes if c != "autre"}
        self.samples: list[tuple[Path, int]] = []
        others: list[Path] = []
        for directory in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            files = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
            if directory.name in wanted:
                target = self.index[wanted[directory.name]]
                self.samples.extend((f, target) for f in files)
            elif "autre" in self.index:
                others.extend(files)
        if others:
            rng = random.Random(seed)
            rng.shuffle(others)
            self.samples.extend((f, self.index["autre"]) for f in others[:per_other])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, target = self.samples[i]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"unreadable: {path}")
        try:
            glyph = preprocess_letter(image)
        except Exception:
            # AHCD strokes routinely touch the tile border, where the
            # frame-removal step erases them; a plain resize keeps the sample.
            glyph = fallback_glyph(image)
        return torch.from_numpy(glyph).unsqueeze(0), target


def pretrain_on_ahcd(model, classes, raw_dir: Path, args, device):
    """Stage 1: fit the whole network, head included, on mapped AHCD glyphs."""
    dataset = AhcdDataset(raw_dir, classes, args.ahcd_other, args.seed)
    if not len(dataset):
        print(f"[WARN] {raw_dir} vide: etape AHCD ignoree")
        return model
    counts = collections.Counter(target for _, target in dataset.samples)
    print(f"[INFO] pre-entrainement AHCD: {len(dataset)} images "
          f"{ {classes[k]: v for k, v in sorted(counts.items())} }")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, args.ahcd_epochs + 1):
        model.train()
        running = correct = seen = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimiser.step()
            running += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += images.size(0)
        print(f"[AHCD {epoch:>2}] loss={running / seen:.4f} acc={correct / seen:.3f}")
    return model


def build_model(classes: list[str], base: Path, device: str) -> tuple[ArabicLetterCNN, bool]:
    model = ArabicLetterCNN(len(classes))
    loaded = False
    if base.is_file():
        checkpoint = torch.load(base, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        # The head is 28-way and class-ordered for AHCD; only the convolutional
        # features transfer.
        features = {k: v for k, v in state.items() if k.startswith("features.")}
        missing = model.load_state_dict(features, strict=False)
        loaded = True
        print(f"[INFO] backbone repris de {base} "
              f"({len(features)} tenseurs, tête réinitialisée pour {len(classes)} classes)")
        if missing.unexpected_keys:
            print(f"[WARN] clés inattendues ignorées: {missing.unexpected_keys}")
    else:
        print(f"[WARN] {base} absent: entraînement depuis une initialisation aléatoire")
    return model.to(device), loaded


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            for truth, guess in zip(targets.numpy(), predictions):
                confusion[truth, guess] += 1
    correct = confusion.trace()
    total = confusion.sum()
    per_class = np.divide(confusion.diagonal(), confusion.sum(1),
                          out=np.zeros(num_classes), where=confusion.sum(1) > 0)
    return correct / max(1, total), per_class, confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the letter CNN on real plate crops.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=2e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--ahcd-dir", type=Path, default=Path("data/arabic_letters/raw"),
                        help="AHCD corpus, one directory per letter.")
    parser.add_argument("--ahcd-epochs", type=int, default=6,
                        help="Stage-1 epochs over AHCD; 0 reuses only the backbone from --base.")
    parser.add_argument("--ahcd-other", type=int, default=1800,
                        help="AHCD glyphs from non-target letters used as stand-in 'autre'.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes, by_split = load_split(args.splits)
    print(f"[INFO] classes: {classes}")
    for split in ("train", "val", "test"):
        counts = collections.Counter(r["label"] for r in by_split[split])
        print(f"[INFO] {split:<5}: {len(by_split[split]):>3} images  {dict(counts)}")
    print(f"[INFO] le jeu de test n'est pas ouvert ici; il est reserve a la phase 3.")

    train_set = RealLetterDataset(by_split["train"], classes, not args.no_augment, args.seed)
    val_set = RealLetterDataset(by_split["val"], classes, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model, _ = build_model(classes, args.base, device)
    if args.ahcd_epochs > 0:
        model = pretrain_on_ahcd(model, classes, args.ahcd_dir, args, device)

    counts = collections.Counter(r["label"] for r in by_split["train"])
    # Inverse-frequency weights: "autre" outnumbers the rarest letter 10:1, and
    # without this the cheapest way to a low loss is to answer "autre" always.
    weights = torch.tensor(
        [len(by_split["train"]) / (len(classes) * max(1, counts[c])) for c in classes],
        dtype=torch.float32, device=device)
    print(f"[INFO] poids de classe: {dict(zip(classes, [round(float(w),2) for w in weights]))}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimiser = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_score, best_state, best_epoch, since = -1.0, None, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimiser.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimiser.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        accuracy, per_class, _ = evaluate(model, val_loader, device, len(classes))
        # Balanced accuracy, not raw accuracy: "autre" is 40 % of the val set
        # and raw accuracy would mostly report how well that one class is done.
        balanced = float(per_class.mean())
        if balanced > best_score:
            best_score, best_epoch, since = balanced, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:>3}] loss={running/len(train_set):.4f} "
                  f"val_acc={accuracy:.3f} val_balanced={balanced:.3f}")
        if since >= args.patience:
            print(f"[INFO] arret anticipe a l'epoque {epoch} (pas d'amelioration depuis {since})")
            break

    model.load_state_dict(best_state)
    accuracy, per_class, confusion = evaluate(model, val_loader, device, len(classes))
    print(f"\n[INFO] meilleur modele: epoque {best_epoch}, val_balanced={best_score:.3f}, val_acc={accuracy:.3f}")
    width = max(len(c) for c in classes) + 2
    print(f"\n{'classe':<{width}}{'n_val':>7}{'rappel':>9}")
    for i, label in enumerate(classes):
        print(f"{label:<{width}}{confusion[i].sum():>7}{per_class[i]:>9.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "labels": classes,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "provenance": {
            "base": str(args.base),
            "ahcd_epochs": args.ahcd_epochs,
            "splits": str(args.splits),
            "val_balanced_accuracy": round(best_score, 4),
            "best_epoch": best_epoch,
            "train_images": len(by_split["train"]),
        },
    }, args.output)
    print(f"\n[INFO] modele enregistre -> {args.output}")


if __name__ == "__main__":
    main()
