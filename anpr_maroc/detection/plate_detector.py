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
        skip_if_cropped: bool = True,
        cropped_aspect_ratio: float = 2.2,
        full_box_area_ratio: float = 0.82,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        # Heuristique "plaque déjà recadrée" : si l'image d'entrée est déjà une
        # plaque (image large et fine) ou si YOLO retourne une boîte couvrant
        # quasiment toute l'image, on ne re-découpe pas la plaque.
        self.skip_if_cropped = skip_if_cropped
        self.cropped_aspect_ratio = cropped_aspect_ratio
        self.full_box_area_ratio = full_box_area_ratio
        self._model = None

    @staticmethod
    def looks_like_cropped_plate(image: np.ndarray, aspect_ratio: float = 2.2) -> bool:
        """Retourne True si l'image ressemble déjà à une plaque recadrée.

        Une plaque marocaine recadrée est nettement plus large que haute
        (ratio largeur/hauteur typiquement > 2). Dans ce cas, il ne faut pas
        chercher une sous-région "plaque" à l'intérieur de la plaque.
        """
        if image is None or image.size == 0:
            return False
        h, w = image.shape[:2]
        if h <= 0:
            return False
        return (w / float(h)) >= aspect_ratio

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
        # 1. Si l'image d'entrée est déjà une plaque recadrée (large et fine),
        #    ne pas re-découper : on renvoie None pour signaler à l'appelant
        #    d'utiliser l'image entière comme plaque.
        if self.skip_if_cropped and self.looks_like_cropped_plate(
            image, self.cropped_aspect_ratio
        ):
            return None

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

        # 2. Si la boîte détectée couvre quasiment toute l'image, c'est que
        #    l'image était déjà (presque) une plaque : on ne re-découpe pas un
        #    morceau partiel de la plaque.
        if self.skip_if_cropped:
            img_h, img_w = image.shape[:2]
            img_area = float(img_w * img_h)
            box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            if img_area > 0 and (box_area / img_area) >= self.full_box_area_ratio:
                return None

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
