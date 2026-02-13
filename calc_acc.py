#!/usr/bin/env python3
"""Compute continual-learning Acc and AAA from Accuracy Matrix in training logs.

Reads "Accuracy Matrix (CNN/NME):" blocks from log files.
Skips incomplete runs (runs that have no Accuracy Matrix).

Matrix format (upper triangular, printed by numpy):
    matrix[i][j] = accuracy of task i at stage j  (j >= i, else 0)
    - Row i  : task i's accuracy across stages i..N-1
    - Col j  : all tasks' accuracies at stage j  (rows 0..j)

Metrics:
    stage_acc[j] = mean of column j  (average accuracy at stage j)
    Acc = stage_acc[N-1]             (final average accuracy)
    AAA = mean(stage_acc)            (average accuracy over all stages)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import List

# ── regex helpers ──────────────────────────────────────────────────────────────
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MATRIX_HEADER_RE = re.compile(r"Accuracy Matrix \((\w+)\):")


# ── matrix parser ─────────────────────────────────────────────────────────────
def parse_accuracy_matrices(text: str, stream: str) -> List[List[List[float]]]:
    """Extract all Accuracy Matrix blocks for the given stream (CNN / NME).

    Returns a list of matrices, where each matrix is a list of rows (list of floats).
    Only fully-printed matrices (ending with ']]') are returned.
    """
    matrices: List[List[List[float]]] = []
    lines = text.splitlines()
    i = 0
    target = stream.upper()

    while i < len(lines):
        m = MATRIX_HEADER_RE.search(lines[i])
        if m and m.group(1).upper() == target:
            rows: List[List[float]] = []
            i += 1
            while i < len(lines):
                row_line = lines[i].strip()
                if not (row_line.startswith("[") or row_line.startswith("[")):
                    break  # left the matrix block
                nums = [float(x) for x in FLOAT_RE.findall(row_line)]
                if nums:
                    rows.append(nums)
                is_last = row_line.rstrip().endswith("]]")
                i += 1
                if is_last:
                    break
            if rows:
                matrices.append(rows)
        else:
            i += 1

    return matrices


# ── metrics computation ───────────────────────────────────────────────────────
def metrics_from_matrix(matrix: List[List[float]]) -> dict:
    """Compute Acc, AAA, and per-stage averages from an accuracy matrix.

    Stage j accuracy = mean of column j over rows 0..j (non-zero entries).
    Acc  = stage_acc[N-1]   (final stage)
    AAA  = mean(stage_acc)  (average over all stages)
    """
    num_tasks = len(matrix)
    stage_acc: List[float] = []

    for j in range(num_tasks):
        col_vals = [
            matrix[i][j]
            for i in range(j + 1)
            if j < len(matrix[i]) and matrix[i][j] > 0
        ]
        if col_vals:
            stage_acc.append(mean(col_vals))

    return {
        "tasks": num_tasks,
        "stage_acc": stage_acc,
        "acc": stage_acc[-1] if stage_acc else 0.0,
        "aaa": mean(stage_acc) if stage_acc else 0.0,
    }


def is_matrix_complete(matrix: List[List[float]]) -> bool:
    """A complete matrix has a positive diagonal (each task was trained)."""
    return all(
        i < len(matrix[i]) and matrix[i][i] > 0
        for i in range(len(matrix))
    )


# ── run aggregation ───────────────────────────────────────────────────────────
def compute_runs(text: str, stream: str) -> List[dict]:
    """Parse matrices, skip incomplete, return per-run metrics."""
    matrices = parse_accuracy_matrices(text, stream)
    runs: List[dict] = []
    for matrix in matrices:
        if not is_matrix_complete(matrix):
            continue
        runs.append(metrics_from_matrix(matrix))
    return runs


def summarize_runs(runs: List[dict]) -> dict:
    accs = [r["acc"] for r in runs]
    aaas = [r["aaa"] for r in runs]
    out = {
        "num_runs": len(runs),
        "acc_mean": mean(accs),
        "aaa_mean": mean(aaas),
        "acc_std": 0.0,
        "aaa_std": 0.0,
    }
    if len(runs) > 1:
        out["acc_std"] = stdev(accs)
        out["aaa_std"] = stdev(aaas)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Acc / AAA from Accuracy Matrix in CL logs."
    )
    parser.add_argument("logs", nargs="+", help="Path(s) to log file(s)")
    parser.add_argument(
        "--stream", default="cnn", choices=["cnn", "nme"],
        help="Metric stream to parse (default: cnn)",
    )
    parser.add_argument(
        "--digits", type=int, default=2,
        help="Decimal places in text output (default: 2)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = []
    for log_path in args.logs:
        path = Path(log_path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        runs = compute_runs(text, args.stream)
        if not runs:
            print(f"WARNING: no complete Accuracy Matrix ({args.stream.upper()}) "
                  f"found in {path}, skipped.")
            continue
        results.append({
            "log": str(path),
            "stream": args.stream.upper(),
            "runs": runs,
            "summary": summarize_runs(runs),
        })

    if not results:
        return

    if args.json:
        print(json.dumps(
            results if len(results) > 1 else results[0],
            indent=2, ensure_ascii=False,
        ))
        return

    fmt = f"{{:.{args.digits}f}}"
    for item in results:
        print(f"log: {item['log']}")
        print(f"stream: {item['stream']}")
        for idx, run in enumerate(item["runs"], start=1):
            print(
                f"  run {idx}: tasks={run['tasks']}  "
                f"Acc={fmt.format(run['acc'])}  "
                f"AAA={fmt.format(run['aaa'])}"
            )
        summary = item["summary"]
        if summary["num_runs"] > 1:
            print(
                f"  => Acc={fmt.format(summary['acc_mean'])}±{fmt.format(summary['acc_std'])}  "
                f"AAA={fmt.format(summary['aaa_mean'])}±{fmt.format(summary['aaa_std'])}"
            )
        else:
            print(
                f"  => Acc={fmt.format(summary['acc_mean'])}  "
                f"AAA={fmt.format(summary['aaa_mean'])}"
            )
        print()


if __name__ == "__main__":
    main()
