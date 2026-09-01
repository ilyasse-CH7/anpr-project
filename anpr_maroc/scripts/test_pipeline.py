"""Script d'inférence bout en bout pour le pipeline ANPR Maroc (Détection YOLO + Segmentation + OCR).

Usage:
    python -m anpr_maroc.scripts.test_pipeline --image data/sample_plates/0.jpeg
    python -m anpr_maroc.scripts.test_pipeline --image data/sample_plates/3.png
    python -m anpr_maroc.scripts.test_pipeline --image data/sample_plates/cx.jpeg --conf 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from anpr_maroc.detection.plate_detector import detect_plate
from anpr_maroc.ocr.easyocr_reader import EasyOCRReader
from anpr_maroc.processing.segmenter import segment_plate_by_layout


def run_pipeline_on_image(
    image_path: Path,
    model_path: str = "models/plate_detector.pt",
    conf_threshold: float = 0.25,
    reader: EasyOCRReader | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Exécute la chaîne complète de traitement sur une image donnée."""
    if reader is None:
        reader = EasyOCRReader()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    h, w = image.shape[:2]
    aspect_ratio = w / float(h) if h > 0 else 1.0

    print(f"\n=======================================================")
    print(f"Traitement de l'image : {image_path.name} ({w}x{h} px)")
    print(f"=======================================================")

    # 1. Étape Détection YOLO
    bbox = None
    try:
        bbox = detect_plate(image, model_path=model_path, conf_threshold=conf_threshold)
    except Exception as e:
        print(f"[-] Avertissement lors de la détection YOLO : {e}")

    is_already_cropped = False
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        print(f"[+] Plaque localisée par YOLO : BBox = ({x1}, {y1}, {x2}, {y2})")
        plate_crop = image[y1:y2, x1:x2]
    else:
        # Si YOLO ne trouve pas de boîte, l'image est peut-être déjà une plaque recadrée (ex: 3.png, 8.jpg)
        print(f"[?] Aucune plaque détectée par YOLO avec conf={conf_threshold}.")
        print(f"[*] Tentative de lecture directe sur l'image entière (cas plaque déjà recadrée)...")
        plate_crop = image
        is_already_cropped = True
        x1, y1, x2, y2 = 0, 0, w, h

    # 2. Étape OCR et Segmentation
    ocr_result = reader.read_and_parse(
        plate_crop,
        try_segment=lambda img: segment_plate_by_layout(img),
    )

    parsed = ocr_result.get("parsed", {})
    serie = parsed.get("left", "")
    letter = parsed.get("letter", "")
    region = parsed.get("right", "")
    matricule = f"{serie} | {letter} | {region}" if (serie or letter or region) else ocr_result.get("raw_text", "")

    print(f"\n--- RÉSULTATS DE LECTURE (OCR) ---")
    print(f" • Numéro de Série (Gauche)   : {serie if serie else '(non détecté)'}")
    print(f" • Lettre Arabe     (Centre)   : {letter if letter else '(non détectée)'}")
    print(f" • Code Région      (Droite)   : {region if region else '(non détecté)'}")
    print(f" • MATRICULE COMPLET          : {matricule if matricule else '(aucun texte reconnu)'}")
    print(f" • Texte brut extrait          : {ocr_result.get('raw_text', '')}")

    # 3. Sauvegarde de la visualisation
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated = image.copy()
        if not is_already_cropped and bbox is not None:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Matricule: {matricule}" if matricule else "Plaque"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        out_path = output_dir / f"result_{image_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"\n[+] Image résultat enregistrée : {out_path}")

    return {
        "image": image_path.name,
        "bbox": bbox,
        "parsed": parsed,
        "matricule": matricule,
        "raw_text": ocr_result.get("raw_text", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline complet ANPR Maroc (Détection + OCR).")
    parser.add_argument(
        "--image",
        type=str,
        help="Chemin vers une image spécifique (ex: data/sample_plates/3.png ou data/sample_plates/0.jpeg)",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="Liste d'images à traiter en série.",
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
        "--output",
        type=str,
        default="data/pipeline_output",
        help="Dossier de sauvegarde des visualisations.",
    )
    args = parser.parse_args()

    image_list = []
    if args.image:
        image_list.append(Path(args.image))
    elif args.images:
        image_list.extend([Path(p) for p in args.images])
    else:
        # Par défaut, tester quelques images du dossier sample_plates
        default_samples = ["data/sample_plates/0.jpeg", "data/sample_plates/3.png", "data/sample_plates/8.jpg"]
        image_list.extend([Path(p) for p in default_samples if Path(p).exists()])

    if not image_list:
        print("Aucune image spécifiée. Utilisez --image <chemin_image>")
        return

    reader = EasyOCRReader()
    output_dir = Path(args.output)

    for img_p in image_list:
        if not img_p.exists():
            print(f"[-] Fichier introuvable : {img_p}")
            continue
        run_pipeline_on_image(
            image_path=img_p,
            model_path=args.model,
            conf_threshold=args.conf,
            reader=reader,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
