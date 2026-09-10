"""Compare letter classifiers on a frozen split, class by class.

Reports, per model: per-letter recall, the confusion matrix, and the mean
confidence of correct versus incorrect predictions -- the last one matters
because the pipeline gates on confidence, so a model that is wrong *and*
confident is worse than one that is wrong and knows it.

The three models do not share a label set, so a comparison needs an explicit
mapping rather than a silent one:

  - the 3-class model emits أ/ب/ه and can never answer د;
  - the AHCD model emits the 28 handwritten classes, which contain ا but no أ,
    because AHCD does not distinguish the hamza.  On a Moroccan plate the bare
    alif and the hamza-alif are the same field, so ا is accepted as أ.

A class a model cannot emit is scored as a miss and flagged in the output;
folding it away would silently credit a model for a letter it does not know.

    python -m anpr_maroc.scripts.evaluate_letter_models --split val
    python -m anpr_maroc.scripts.evaluate_letter_models --split test   # phase 3
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from anpr_maroc.ocr.arabic_letter_model import ArabicLetterClassifier

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = Path("data/arabic_letters/splits.json")
OTHER = "autre"

# AHCD has no hamza-alif; on a plate the two are the same field.
EQUIVALENT = {"ا": "أ"}

DEFAULT_MODELS = {
    "3-classes (actuel)": "models/arabic_letter_classifier_finetuned.pt",
    "ahcd (28 classes)": "models/arabic_letter_classifier_ahcd.pt",
    "reel v1 (backbone ahcd)": "models/arabic_letter_classifier_real.pt",
    "reel v2 (pre-entr. 4 classes)": "models/arabic_letter_classifier_real2.pt",
}


def normalise(label: str) -> str:
    return EQUIVALENT.get(label, label)


def evaluate(model_path: Path, records, classes, reject_threshold: float):
    classifier = ArabicLetterClassifier(model_path)
    if not classifier.available:
        return None
    emits = {normalise(l) for l in classifier.labels}
    confusion = collections.Counter()
    confidences = {"ok": [], "ko": []}
    unreachable = sorted(c for c in classes if c != OTHER and c not in emits)

    for record in records:
        image = cv2.imread(str(PROJECT / record["path"]))
        raw_label, score = classifier.predict(image)
        predicted = normalise(raw_label) if raw_label else OTHER
        # A model without an explicit reject class expresses "not a letter"
        # only through low confidence, so give it that route.
        if OTHER not in emits and score < reject_threshold:
            predicted = OTHER
        truth = record["label"]
        confusion[(truth, predicted)] += 1
        confidences["ok" if predicted == truth else "ko"].append(score)

    return {
        "labels": classifier.labels,
        "confusion": confusion,
        "confidences": confidences,
        "unreachable": unreachable,
    }


def report(name: str, result, classes, letters_only: bool) -> dict:
    confusion = result["confusion"]
    seen = sorted({p for _, p in confusion} | set(classes), key=lambda c: (c == OTHER, c))
    scored = [c for c in classes if c != OTHER] if letters_only else classes

    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    if result["unreachable"]:
        print(f"  ne peut pas emettre : {', '.join(result['unreachable'])} "
              f"(comptees comme erreurs)")

    width = max(len(c) for c in seen) + 2
    # Recall alone flatters a model that answers one class to everything: the
    # 3-class model scores a perfect recall on أ precisely because it says أ
    # to most of the "autre" bucket too.  Precision is what exposes that.
    print(f"\n  {'verite':<{width}}{'n':>4}{'rappel':>9}{'prec.':>8}   predictions")
    recalls = {}
    total_correct = total = 0
    for truth in scored:
        row = {p: n for (t, p), n in confusion.items() if t == truth}
        n = sum(row.values())
        correct = row.get(truth, 0)
        if n:
            recalls[truth] = correct / n
            total_correct += correct
            total += n
        predicted_as = sum(c for (t, p), c in confusion.items() if p == truth)
        precision = correct / predicted_as if predicted_as else float("nan")
        detail = ", ".join(f"{p}:{c}" for p, c in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {truth:<{width}}{n:>4}{(correct / n if n else 0):>9.3f}{precision:>8.3f}   {detail}")

    macro = float(np.mean(list(recalls.values()))) if recalls else 0.0
    micro = total_correct / total if total else 0.0
    ok, ko = result["confidences"]["ok"], result["confidences"]["ko"]
    print(f"\n  rappel macro : {macro:.3f}   |   exactitude : {micro:.3f}  ({total_correct}/{total})")
    print(f"  confiance moyenne  correct: {np.mean(ok) if ok else 0:.3f} (n={len(ok)})"
          f"   incorrect: {np.mean(ko) if ko else 0:.3f} (n={len(ko)})")
    return {"macro_recall": macro, "accuracy": micro, "per_class": recalls}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare letter classifiers on a frozen split.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--letters-only", action="store_true",
                        help="Score only the real letters, ignoring the 'autre' bucket.")
    parser.add_argument("--reject-threshold", type=float, default=0.5,
                        help="Confidence below which a model with no 'autre' class is read as a reject.")
    parser.add_argument("--models", nargs="*", default=None, help="name=path pairs; defaults to the three project models.")
    args = parser.parse_args()

    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    records = [r for r in payload["records"] if r["split"] == args.split]
    classes = payload["classes"]
    if args.letters_only:
        records = [r for r in records if r["label"] != OTHER]

    counts = collections.Counter(r["label"] for r in records)
    print(f"jeu : {args.split}  ({len(records)} images)  {dict(counts)}")
    if args.split == "test":
        print("NOTE: jeu de test - a n'utiliser qu'une fois, pour la decision finale.")

    models = DEFAULT_MODELS
    if args.models:
        models = dict(pair.split("=", 1) for pair in args.models)

    summary = {}
    for name, path in models.items():
        result = evaluate(Path(path), records, classes, args.reject_threshold)
        if result is None:
            print(f"\n[SKIP] {name}: {path} introuvable ou illisible")
            continue
        summary[name] = report(name, result, classes, args.letters_only)

    if summary:
        print(f"\n{'=' * 66}\nRECAPITULATIF ({args.split})\n{'=' * 66}")
        width = max(len(n) for n in summary) + 2
        print(f"{'modele':<{width}}{'macro':>8}{'exact.':>9}")
        for name, values in sorted(summary.items(), key=lambda kv: -kv[1]["macro_recall"]):
            print(f"{name:<{width}}{values['macro_recall']:>8.3f}{values['accuracy']:>9.3f}")


if __name__ == "__main__":
    main()
