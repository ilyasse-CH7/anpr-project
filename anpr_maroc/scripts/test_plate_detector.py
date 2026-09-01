"""Script de test d'intégration pour le détecteur de plaques (PlateDetector).

Usage:
    python -m anpr_maroc.scripts.test_plate_detector --images data/sample_plates/0.jpeg data/sample_plates/cx.jpeg data/sample_plates/kk.jpeg --output data/detections_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from anpr_maroc.detection.plate_detector import detect_plate


def main() -> None:
    parser = argparse.ArgumentParser(description="Test d'intégration du détecteur de plaques d'immatriculation.")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["data/sample_plates/0.jpeg", "data/sample_plates/cx.jpeg", "data/sample_plates/kk.jpeg"],
        help="Liste des chemins d'images de véhicules complets à tester.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/plate_detector.pt",
        help="Chemin vers le modèle YOLO entraîné.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/detections_output",
        help="Dossier de sauvegarde des images avec bounding boxes dessinées.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Seuil de confiance de détection.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Test d'intégration de détection de plaque ===")
    print(f"Modèle : {args.model}")
    print(f"Seuil de confiance : {args.conf}")
    print(f"Dossier de sortie : {output_dir}\n")

    for img_path_str in args.images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[-] Image introuvable : {img_path}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[-] Impossible de lire l'image : {img_path}")
            continue

        try:
            bbox = detect_plate(image, model_path=args.model, conf_threshold=args.conf)
        except Exception as err:
            print(f"[-] Erreur lors de l'inférence sur {img_path.name} : {err}")
            continue

        annotated_image = image.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                annotated_image,
                "License Plate",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            print(f"[+] Plaque détectée sur {img_path.name} -> BBox: ({x1}, {y1}, {x2}, {y2})")
        else:
            print(f"[?] Aucune plaque détectée sur {img_path.name}")

        out_file = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(out_file), annotated_image)
        print(f"    Visualisation sauvegardée : {out_file}")


if __name__ == "__main__":
    main()
