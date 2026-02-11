#!/usr/bin/env python3
"""Render a Markdown report summarizing optimization comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "trustworthiness_delta": 0.005,
    "knn_overlap": 0.85,
    "pairwise_corr": 0.99,
}


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def format_pass(entry: dict[str, object]) -> str:
    delta = abs(entry["trustworthiness_delta"])
    overlap = entry["knn_overlap"]
    corr = entry["pairwise_corr"]
    passed = (
        delta <= THRESHOLDS["trustworthiness_delta"]
        and overlap >= THRESHOLDS["knn_overlap"]
        and corr >= THRESHOLDS["pairwise_corr"]
    )
    return "✅" if passed else "❌"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("benchmarks/summary.json"),
        help="JSON summary created by compare_benchmark_results.py",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("benchmarks/report.md"),
        help="Markdown report destination",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary_file)
    lines = [
        "# UMATO Optimization Report",
        "",
        "Comparison uses deterministic vs fast execution on the sample scenario "
        "(n_neighbors=15) described in `benchmarks/similarity/`.",
        "",
        "| Dataset | n_jobs | Trustworthiness Δ | KNN Overlap | Pairwise Corr | Fast Speedup | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for entry in sorted(summary, key=lambda e: (e["dataset"], e["n_jobs"])):
        lines.append(
            "| {dataset} | {n_jobs} | {delta:.4f} | {overlap:.3f} | {corr:.4f} | {speedup:.2f} | {pass_flag} |".format(
                dataset=entry["dataset"],
                n_jobs=entry["n_jobs"],
                delta=entry["trustworthiness_delta"],
                overlap=entry["knn_overlap"],
                corr=entry["pairwise_corr"],
                speedup=entry["fast_speedup"],
                pass_flag=format_pass(entry),
            )
        )

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
