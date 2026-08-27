"""Inference utilities for the dedicated Moroccan-plate Arabic letter model.

The model deliberately receives an isolated glyph, not a full line of Arabic
text.  Its preprocessing keeps all connected components of the glyph, including
the hamza/madda above or below an alif; this is the information discarded by the
former ``matchShapes`` fallback.
"""

from __future__ import annotations

from pathlib import Path
import typing as t

import cv2
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Makes importing the OCR package safe without torch.
    torch = None
    nn = None


IMAGE_SIZE = 96
DEFAULT_MODEL_PATH = Path("models/arabic_letter_classifier.pt")


def preprocess_letter(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Return a normalized glyph image while retaining detached diacritics.

    Plate frames and separator bars touch an image edge.  They are discarded,
    while detached interior components (hamza, madda and dots) are retained.
    """
    if image is None or image.size == 0:
        raise ValueError("empty letter image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    min_area = max(3, int(binary.size * 0.00008))
    height, width = binary.shape
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        touches_edge = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        if area >= min_area and not touches_edge:
            clean[labels == component] = 255

    ys, xs = np.where(clean > 0)
    if xs.size == 0:
        raise ValueError("no glyph pixels after removing plate frame")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    glyph = clean[y0:y1 + 1, x0:x1 + 1]
    pad = max(6, int(max(glyph.shape) * 0.18))
    canvas = np.zeros((glyph.shape[0] + 2 * pad, glyph.shape[1] + 2 * pad), dtype=np.uint8)
    canvas[pad:pad + glyph.shape[0], pad:pad + glyph.shape[1]] = glyph
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class ArabicLetterCNN(nn.Module if nn is not None else object):
    """Small CNN suitable for 96×96 single-glyph plate crops."""

    def __init__(self, num_classes: int):
        if nn is None:
            raise RuntimeError("PyTorch is required for ArabicLetterCNN")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.classifier(self.features(x))


class ArabicLetterClassifier:
    """Load a trained checkpoint and predict one plate letter.

    A missing checkpoint is an expected state during data collection and returns
    an empty prediction, allowing the caller to keep its existing OCR fallback.
    """

    def __init__(self, model_path: t.Union[str, Path] = DEFAULT_MODEL_PATH):
        self.path = Path(model_path)
        self.model: t.Optional[ArabicLetterCNN] = None
        self.labels: list[str] = []
        if torch is None or not self.path.is_file():
            return
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        self.labels = list(checkpoint["labels"])
        self.model = ArabicLetterCNN(len(self.labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @property
    def available(self) -> bool:
        return self.model is not None and bool(self.labels)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        if not self.available or torch is None:
            return "", 0.0
        try:
            glyph = preprocess_letter(image)
            tensor = torch.from_numpy(glyph).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            score, index = torch.max(probabilities, dim=0)
            return self.labels[int(index.item())], float(score.item())
        except (ValueError, cv2.error):
            return "", 0.0
