"""Module de détection de plaques d'immatriculation via YOLO (Ultralytics)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class PlateDetector:
    """Détecteur de plaque d'immatriculation basé sur YOLOv8."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/plate_detector.pt",
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as err:
                raise ImportError(
                    "Le package 'ultralytics' est requis pour la détection. "
                    "Installez-le avec `pip install ultralytics`."
                ) from err

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Modèle introuvable à l'emplacement : {self.model_path}. "
                    "Veuillez entraîner ou placer les poids YOLO dans le dossier 'models/'."
                )
            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Détecte la plaque d'immatriculation dans l'image.

        Args:
            image: Image d'entrée (BGR sous format numpy.ndarray).

        Returns:
            Tuple (x1, y1, x2, y2) des coordonnées de la bounding box
            avec le score de confiance le plus élevé, ou None si aucune plaque n'est détectée.
        """
        model = self._get_model()
        results = model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return None

        # Sélectionner la bounding box avec la plus haute confiance
        boxes = results[0].boxes
        best_box_idx = int(boxes.conf.argmax())
        coords = boxes.xyxy[best_box_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = map(int, coords[:4])
        return int(x1), int(y1), int(x2), int(y2)


_default_detector: Optional[PlateDetector] = None


def detect_plate(
    image: np.ndarray,
    model_path: Union[str, Path] = "models/plate_detector.pt",
    conf_threshold: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """Détecte la plaque dans une image et retourne les coordonnées de la bounding box (x1, y1, x2, y2).

    Args:
        image: Image d'entrée (BGR sous format numpy.ndarray).
        model_path: Chemin vers le modèle YOLO entraîné.
        conf_threshold: Seuil minimal de confiance.

    Returns:
        Tuple (x1, y1, x2, y2) ou None si rien n'est détecté.
    """
    global _default_detector
    if _default_detector is None or _default_detector.model_path != Path(model_path):
        _default_detector = PlateDetector(model_path=model_path, conf_threshold=conf_threshold)
    return _default_detector.detect(image)
