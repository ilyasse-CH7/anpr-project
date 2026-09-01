"""Tests pour le module de détection de plaques."""

import unittest
from pathlib import Path
import numpy as np

from anpr_maroc.detection.plate_detector import PlateDetector, detect_plate


class TestPlateDetector(unittest.TestCase):
    """Tests unitaires pour PlateDetector."""

    def test_detector_initialization(self) -> None:
        detector = PlateDetector(model_path="models/plate_detector.pt")
        self.assertEqual(detector.model_path, Path("models/plate_detector.pt"))
        self.assertEqual(detector.conf_threshold, 0.25)

    def test_detect_on_blank_image(self) -> None:
        blank_image = np.zeros((400, 600, 3), dtype=np.uint8)
        # Sur une image noire vierge, aucune plaque ne doit être détectée
        bbox = detect_plate(blank_image, model_path="models/plate_detector.pt")
        self.assertIsNone(bbox)


if __name__ == "__main__":
    unittest.main()
