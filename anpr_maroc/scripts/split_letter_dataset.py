"""Freeze a leakage-free train/val/test split over the annotated plate crops.

The unit of splitting is the burst group from ``queue_manifest.jsonl``, not the
image: consecutive frames of one car are near-identical, so putting two of them
on opposite sides of the boundary would let the test set measure memorisation
instead of generalisation.

Groups are multi-class -- a 10-second burst in a car park catches several cars,
and the grouping deliberately errs towards merging -- so a group cannot be
assigned per class.  Each group is placed once, globally, by a greedy pass that
sends it to whichever split is currently furthest below its target for the
classes that group actually contains.

    python -m anpr_maroc.scripts.split_letter_dataset

The result is written to ``data/arabic_letters/splits.json`` and is meant to be
generated once and then left alone.  Re-running it after the test set has been
looked at silently invalidates every number measured against it.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

LETTERS = Path("data/arabic_letters")
DEFAULT_MANIFEST = LETTERS / "queue_manifest.jsonl"
DEFAULT_ANNOTATIONS = LETTERS / "annotations.jsonl"
DEFAULT_OUTPUT = LETTERS / "splits.json"

REJECT_LABEL = "_reject"
OTHER_CLASS = "autre"


def load_records(manifest: Path, annotations: Path, min_per_class: int, keep_below: bool):
    groups = {
        json.loads(line)["file"]: json.loads(line)["group"]
        for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    records = []
    for line in annotations.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["file"] not in groups:
            print(f"[WARN] {entry['file']}: absent from the manifest, skipped")
            continue
        label = OTHER_CLASS if entry["label"] == REJECT_LABEL else entry["label"]
        records.append({
            "file": entry["file"],
            "path": entry["path"],
            "label": label,
            "group": groups[entry["file"]],
        })

    counts = collections.Counter(record["label"] for record in records)
    dropped = {
        label for label, n in counts.items()
        if label != OTHER_CLASS and n < min_per_class and not keep_below
    }
    for label in sorted(dropped):
        print(f"[DROP] {label}: {counts[label]} images, below --min-per-class={min_per_class}")
    return [r for r in records if r["label"] not in dropped], counts, dropped


def assign(records, ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Greedily place each group into the split that most needs its classes."""
    per_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        per_group[record["group"]][record["label"]] += 1
    totals = collections.Counter(record["label"] for record in records)
    target = {
        split: {label: totals[label] * ratio for label in totals}
        for split, ratio in ratios.items()
    }
    have = {split: collections.Counter() for split in ratios}

    # Largest groups first: they constrain the balance most, and placing them
    # while every split is still empty leaves the small groups to correct the
    # residual skew.  The shuffle only breaks ties between equal-sized groups.
    order = sorted(per_group, key=lambda g: -sum(per_group[g].values()))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -sum(per_group[g].values()))

    placement: dict[str, str] = {}
    for group in order:
        counts = per_group[group]
        best, best_score = None, None
        for split in ratios:
            # Deficit the group would fill, normalised per class so a rare
            # letter outweighs the abundant "autre" bucket.
            score = sum(
                min(counts[label], max(0.0, target[split][label] - have[split][label]))
                / max(1.0, totals[label])
                for label in counts
            )
            if best_score is None or score > best_score:
                best, best_score = split, score
        placement[group] = best
        have[best].update(counts)
    return placement


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a grouped train/val/test split.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", type=float, default=0.55)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.25)
    parser.add_argument("--min-per-class", type=int, default=15, help="Classes with fewer real images than this are dropped.")
    parser.add_argument("--keep-below-threshold", action="store_true", help="Keep under-supported classes anyway (their metrics stay indicative).")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split. This invalidates every number already measured on the old test set.")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"{args.output} already exists.\n"
            "A split is meant to be frozen once. Re-run with --force only if you accept "
            "that all previously reported test numbers become meaningless."
        )

    ratios = {"train": args.train, "val": args.val, "test": args.test}
    total = sum(ratios.values())
    ratios = {split: value / total for split, value in ratios.items()}

    records, counts, dropped = load_records(
        args.manifest, args.annotations, args.min_per_class, args.keep_below_threshold
    )
    placement = assign(records, ratios, args.seed)
    for record in records:
        record["split"] = placement[record["group"]]

    classes = sorted({r["label"] for r in records})
    table = {split: collections.Counter() for split in ratios}
    group_count = collections.Counter()
    for record in records:
        table[record["split"]][record["label"]] += 1
    for group, split in placement.items():
        group_count[split] += 1

    width = max(len(c) for c in classes) + 2
    print(f"\n{'classe':<{width}}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    print("-" * (width + 26))
    for label in classes:
        row = [table[s][label] for s in ("train", "val", "test")]
        print(f"{label:<{width}}{row[0]:>7}{row[1]:>6}{row[2]:>6}{sum(row):>7}")
    print("-" * (width + 26))
    print(f"{'images':<{width}}" + "".join(f"{sum(table[s].values()):>7}" if s == 'train'
          else f"{sum(table[s].values()):>6}" for s in ("train", "val", "test"))
          + f"{len(records):>7}")
    print(f"{'groupes':<{width}}{group_count['train']:>7}{group_count['val']:>6}{group_count['test']:>6}{len(placement):>7}")

    payload = {
        "seed": args.seed,
        "ratios": ratios,
        "min_per_class": args.min_per_class,
        "classes": classes,
        "dropped_classes": {label: counts[label] for label in sorted(dropped)},
        "counts": {split: dict(table[split]) for split in ratios},
        "records": records,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] split figé -> {args.output}")

    thin = [l for l in classes if l != OTHER_CLASS and table["test"][l] < 8]
    if thin:
        print("[WARN] classes dont le jeu de test est trop petit pour une precision fiable: "
              + ", ".join(f"{l} (n={table['test'][l]})" for l in thin))


if __name__ == "__main__":
    main()
