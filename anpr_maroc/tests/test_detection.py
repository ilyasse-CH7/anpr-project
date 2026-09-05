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

    def test_looks_like_cropped_plate(self) -> None:
        # Image large et fine (ratio ~4.5) = plaque déjà recadrée
        wide_plate = np.zeros((100, 450, 3), dtype=np.uint8)
        self.assertTrue(PlateDetector.looks_like_cropped_plate(wide_plate))
        # Image de véhicule (quasi carrée) = pas une plaque recadrée
        vehicle = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertFalse(PlateDetector.looks_like_cropped_plate(vehicle))

    def test_cropped_plate_skips_yolo(self) -> None:
        # Une plaque déjà recadrée ne doit pas être re-découpée par YOLO.
        detector = PlateDetector(model_path="models/plate_detector.pt")
        wide_plate = np.zeros((100, 450, 3), dtype=np.uint8)
        self.assertIsNone(detector.detect(wide_plate))


if __name__ == "__main__":
    unittest.main()
