"""Aggregate OdinW incremental eval logs into a CSV matrix."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.eval import EVAL_METRIC_CHOICES
from scripts.odinw.odinw_datasets import ODINW_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="mAP50-95",
        choices=list(EVAL_METRIC_CHOICES) + ["precision50", "recall50"],
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Flat layout: eval_on_<dataset>/logs/test_log.jsonl (zero-shot).",
    )
    return parser.parse_args()


def read_metric(log_path: Path, metric: str) -> Optional[float]:
    if not log_path.is_file():
        return None
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    data = json.loads(line)
    test_metrics = data.get("test_metrics", data)
    if metric not in test_metrics:
        raise KeyError(f"Metric {metric!r} not in {log_path}: keys={list(test_metrics.keys())}")
    return float(test_metrics[metric])


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    dataset_names = [ds.name for ds in ODINW_DATASETS]

    rows: list = []

    if args.flat:
        row_metrics = {}
        for name in dataset_names:
            log_path = eval_root / f"eval_on_{name}" / "logs" / "test_log.jsonl"
            row_metrics[name] = read_metric(log_path, args.metric)
        rows.append(("zero-shot", row_metrics))
    else:
        stage_dirs = sorted(
            [p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith("stage_")],
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
        for stage_dir in stage_dirs:
            stage_label = stage_dir.name
            row_metrics = {}
            for name in dataset_names:
                log_path = stage_dir / f"eval_on_{name}" / "logs" / "test_log.jsonl"
                row_metrics[name] = read_metric(log_path, args.metric)
            rows.append((stage_label, row_metrics))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage"] + dataset_names)
        for stage_label, metrics in rows:
            writer.writerow(
                [stage_label] + [metrics.get(name) for name in dataset_names]
            )

    print(f"Wrote {output_path} ({len(rows)} rows x {len(dataset_names)} datasets)")


if __name__ == "__main__":
    main()
