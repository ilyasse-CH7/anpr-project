"""Prépare (semi-automatiquement) un dataset de lettres arabes pour le CNN
à partir des images du dataset YOLO (`data/plate_detection`).

POURQUOI cet outil ?
--------------------
Le dataset YOLO ne contient QUE des labels de localisation de plaque
(classe `0: LP`). Il n'indique NULLE PART quelle est la lettre arabe centrale.
On ne peut donc PAS entraîner directement le CNN de lettres avec ces labels.

Ce script comble le manque : pour chaque image de véhicule/plaque, il
1. détecte + recadre la plaque avec YOLO (sauf si l'image est déjà une plaque),
2. segmente la zone centrale (la lettre arabe),
3. enregistre ce crop de lettre dans un dossier `to_label/`.

Il reste ensuite UNE seule étape MANUELLE : trier ces crops dans des
sous-dossiers nommés par la lettre (ex: `ب/`, `ه/`, `م/`, `أ/`, ...).
Le dossier ainsi trié devient l'entrée de `train_arabic_letter_classifier.py`.

Usage:
    python -m anpr_maroc.scripts.build_arabic_letter_dataset \
      --images data/plate_detection/train/images \
      --output data/arabic_letters/to_label
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from anpr_maroc.detection.plate_detector import detect_plate
from anpr_maroc.processing.segmenter import segment_plate_by_layout


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_images(images_dir: Path):
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraire les crops de lettre arabe centrale depuis un dossier d'images (dataset YOLO)."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Dossier d'images sources (ex: data/plate_detection/train/images).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Dossier de sortie où seront écrits les crops de lettre à trier manuellement.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/plate_detector.pt",
        help="Chemin vers les poids YOLO.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Seuil de confiance de détection YOLO.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Nombre maximum d'images à traiter (0 = toutes).",
    )
    args = parser.parse_args()

    if not args.images.is_dir():
        raise NotADirectoryError(f"Dossier d'images introuvable : {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    processed = 0
    for image_path in _iter_images(args.images):
        if args.limit and processed >= args.limit:
            break
        processed += 1

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[-] Illisible, ignorée : {image_path.name}")
            skipped += 1
            continue

        # 1. Détection + recadrage plaque (None => image déjà une plaque)
        try:
            bbox = detect_plate(image, model_path=args.model, conf_threshold=args.conf)
        except Exception as err:  # noqa: BLE001 - on continue sur l'image suivante
            print(f"[-] Erreur détection sur {image_path.name} : {err}")
            skipped += 1
            continue

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            plate_crop = image[y1:y2, x1:x2]
        else:
            plate_crop = image

        if plate_crop is None or plate_crop.size == 0:
            skipped += 1
            continue

        # 2. Segmentation -> zone centrale = lettre arabe
        try:
            zones = segment_plate_by_layout(plate_crop)
        except Exception as err:  # noqa: BLE001
            print(f"[-] Segmentation échouée sur {image_path.name} : {err}")
            skipped += 1
            continue

        if not zones or len(zones) < 2 or zones[1] is None or zones[1].size == 0:
            skipped += 1
            continue

        out_file = args.output / f"{image_path.stem}_letter.png"
        if not cv2.imwrite(str(out_file), zones[1]):
            print(f"[-] Écriture impossible : {out_file}")
            skipped += 1
            continue
        written += 1

    print(
        f"\nTerminé : {written} crops de lettre écrits dans {args.output} "
        f"(traitées={processed}, ignorées={skipped})."
    )
    print(
        "\nÉTAPE MANUELLE SUIVANTE :\n"
        f"  Triez les images de {args.output} dans des sous-dossiers nommés par la lettre\n"
        "  (ex: data/arabic_letters/raw/ب/, data/arabic_letters/raw/ه/, ...),\n"
        "  puis lancez l'entraînement :\n"
        "    python -m anpr_maroc.scripts.train_arabic_letter_classifier "
        "--data data/arabic_letters/raw"
    )


if __name__ == "__main__":
    main()
