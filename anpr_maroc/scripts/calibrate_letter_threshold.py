"""Choose the pipeline's letter-confidence gate, and report what it costs.

The pipeline emits a letter only above a confidence threshold and otherwise
writes the ``UNKNOWN_LETTER`` sentinel, so that threshold is a real decision:
too low and the plate carries an invented letter, too high and the field is
never filled.

The threshold is swept on **validation** and only read out on test.  Tuning it
on the test split would reintroduce exactly the leakage the dataset was rebuilt
to remove (see docs/arabic_letter_model.md §1), and the test split is small
enough that its high-threshold precision is 3 samples of noise.

    python -m anpr_maroc.scripts.calibrate_letter_threshold
    python -m anpr_maroc.scripts.calibrate_letter_threshold --min-precision 0.5
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from anpr_maroc.ocr.arabic_letter_model import ArabicLetterClassifier

PROJECT = Path(__file__).resolve().parents[2]
OTHER = "autre"
DEFAULT_MODEL = "models/arabic_letter_classifier_real2.pt"


def predict_split(classifier: ArabicLetterClassifier, records, split: str):
    out = []
    for record in records:
        if record["split"] != split:
            continue
        label, score = classifier.predict(cv2.imread(str(PROJECT / record["path"])))
        out.append((record["label"], label or OTHER, score))
    return out


def score_at(predictions, threshold: float) -> dict:
    """Score one operating point the way the pipeline actually behaves.

    Precision/recall are computed over *emitted letters* only: a prediction
    gated back to OTHER is not a wrong reading, it is an abstention, and the
    pipeline shows it as such.
    """
    tp = fp = fn = correct = 0
    per_true, per_emitted = collections.Counter(), collections.Counter()
    for truth, predicted, score in predictions:
        gated = predicted if (predicted != OTHER and score >= threshold) else OTHER
        correct += gated == truth
        if gated != OTHER:
            per_emitted[gated] += 1
            if gated == truth:
                tp += 1
                per_true[gated] += 1
            else:
                fp += 1
        elif truth != OTHER:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": correct / len(predictions),
            "per_true": per_true, "per_emitted": per_emitted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--splits", type=Path, default=Path("data/arabic_letters/splits.json"))
    parser.add_argument("--min-precision", type=float, default=0.60,
                        help="Precision demanded on validation; the highest-recall point reaching it wins.")
    parser.add_argument("--read-test", action="store_true",
                        help="Also read the chosen point out on the sealed test split.")
    args = parser.parse_args()

    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    classifier = ArabicLetterClassifier(args.model)
    if not classifier.available:
        raise SystemExit(f"modele introuvable ou illisible : {args.model}")
    letters = [c for c in classifier.labels if c != OTHER]

    val = predict_split(classifier, payload["records"], "val")
    grid = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
    rows = [score_at(val, t) for t in grid]

    print(f"modele : {args.model}   lettres : {', '.join(letters)}")
    print(f"\nCALIBRATION sur la VALIDATION ({len(val)} images)")
    print(f"{'seuil':>6}{'emises':>8}{'justes':>8}{'prec.':>8}{'rappel':>9}{'F1':>8}{'exact.':>9}")
    for row in rows:
        print(f"{row['threshold']:>6.2f}{row['tp'] + row['fp']:>8}{row['tp']:>8}"
              f"{row['precision']:>8.3f}{row['recall']:>9.3f}{row['f1']:>8.3f}{row['accuracy']:>9.3f}")

    reaching = [r for r in rows if r["precision"] >= args.min_precision and r["tp"]]
    best_f1 = max(rows, key=lambda r: r["f1"])
    if reaching:
        chosen = max(reaching, key=lambda r: r["recall"])
        print(f"\nprecision >= {args.min_precision:.2f} atteinte : seuil {chosen['threshold']:.2f}")
    else:
        # Worth stating plainly rather than silently falling back: no threshold
        # buys usable precision, so raising it only turns the model off.
        ceiling = max(r["precision"] for r in rows)
        print(f"\nAUCUN seuil n'atteint une precision de {args.min_precision:.2f} "
              f"(plafond mesure : {ceiling:.3f}).")
        print("Monter le seuil n'achete pas de la precision, il eteint le modele.")
        # Among the points that still emit every letter, take the best accuracy:
        # past that, the model degenerates into an 'أ'-only detector.
        full_coverage = [r for r in rows if all(r["per_emitted"][l] for l in letters)]
        chosen = max(full_coverage or [best_f1], key=lambda r: r["accuracy"])
        print(f"Point retenu (toutes lettres encore emises, exactitude max) : "
              f"seuil {chosen['threshold']:.2f}")

    print(f"  -> precision {chosen['precision']:.3f}  rappel {chosen['recall']:.3f}  "
          f"F1 {chosen['f1']:.3f}  exactitude 4 classes {chosen['accuracy']:.3f}")

    if args.read_test:
        test = predict_split(classifier, payload["records"], "test")
        row = score_at(test, chosen["threshold"])
        print(f"\nLECTURE sur le TEST ({len(test)} images) au seuil {chosen['threshold']:.2f}"
              "  -- lecture unique, aucun reglage ici")
        print(f"  emises {row['tp'] + row['fp']}  justes {row['tp']}   precision {row['precision']:.3f}   "
              f"rappel {row['recall']:.3f}   exactitude 4 classes {row['accuracy']:.3f}")
        for letter in letters:
            emitted = row["per_emitted"][letter]
            rate = f" = {row['per_true'][letter] / emitted:.3f}" if emitted else ""
            print(f"    {letter} : {row['per_true'][letter]}/{emitted} emissions justes{rate}")

        # The number that matters once segmentation has done its job.
        framed = [(t, p) for t, p, _ in test if t != OTHER]
        good = sum(t == p for t, p in framed)
        print(f"\n  sur zone lettre correctement cadree : {good}/{len(framed)} = "
              f"{good / len(framed):.3f}")


if __name__ == "__main__":
    main()
