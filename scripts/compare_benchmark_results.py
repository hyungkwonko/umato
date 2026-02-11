#!/usr/bin/env python3
"""Compare deterministic and fast UMATO executions using stored benchmark data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def load_results(path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def _compute_knn_overlap(embedding_a: np.ndarray, embedding_b: np.ndarray, k: int = 15) -> float:
    n = embedding_a.shape[0]
    if n <= 1:
        return 1.0
    k = min(k, n - 1)
    nn_a = NearestNeighbors(n_neighbors=k + 1).fit(embedding_a)
    nn_b = NearestNeighbors(n_neighbors=k + 1).fit(embedding_b)
    neighbors_a = nn_a.kneighbors(return_distance=False)[:, 1:]
    neighbors_b = nn_b.kneighbors(return_distance=False)[:, 1:]
    overlap = [
        len(set(row_a).intersection(row_b)) / k
        for row_a, row_b in zip(neighbors_a, neighbors_b)
    ]
    return float(np.mean(overlap))


def _compute_pairwise_corr(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    if embedding_a.shape[0] < 2:
        return 1.0
    da = pdist(embedding_a)
    db = pdist(embedding_b)
    corr = np.corrcoef(da, db)[0, 1]
    if math.isnan(corr):
        return 1.0
    return float(corr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("benchmarks/results.jsonl"),
        help="Source benchmark JSONL file",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("benchmarks/summary.json"),
        help="Summary JSON to write comparison metrics to",
    )
    parser.add_argument(
        "--similarity-dir",
        type=Path,
        default=Path("benchmarks/similarity"),
        help="Location of stored similarity reference data",
    )
    args = parser.parse_args()

    runs = load_results(args.results_file)
    grouped: dict[tuple[str, str, int], dict[str, dict[str, object]]] = {}
    for entry in runs:
        if not entry.get("success") or not entry.get("embedding_path"):
            continue
        key = (entry["dataset"], entry["scenario"], entry["n_jobs"])
        grouped.setdefault(key, {})[entry["execution_mode"]] = entry

    similarity_cache: dict[str, np.ndarray] = {}
    summary = []
    for (dataset, scenario, n_jobs), executions in grouped.items():
        if scenario != "sample":
            continue
        if "deterministic" not in executions or "fast" not in executions:
            continue

        det = executions["deterministic"]
        fast = executions["fast"]
        similarity_ref = det.get("similarity_reference") or fast.get("similarity_reference")
        if similarity_ref is None:
            continue

        if similarity_ref not in similarity_cache:
            with np.load(similarity_ref) as buf:
                similarity_cache[similarity_ref] = buf["data"].astype(np.float32)
        sample_data = similarity_cache[similarity_ref]

        emb_det = np.load(det["embedding_path"])
        emb_fast = np.load(fast["embedding_path"])

        trust_det = trustworthiness(sample_data, emb_det, n_neighbors=15)
        trust_fast = trustworthiness(sample_data, emb_fast, n_neighbors=15)
        trust_delta = trust_fast - trust_det

        overlap = _compute_knn_overlap(emb_det, emb_fast)
        corr = _compute_pairwise_corr(emb_det, emb_fast)

        det_time = det["duration_sec"]
        fast_time = fast["duration_sec"]
        speedup = det_time / fast_time if fast_time > 0 else float("inf")

        entry = {
            "dataset": dataset,
            "n_jobs": n_jobs,
            "trustworthiness_det": trust_det,
            "trustworthiness_fast": trust_fast,
            "trustworthiness_delta": trust_delta,
            "knn_overlap": overlap,
            "pairwise_corr": corr,
            "det_duration": det_time,
            "fast_duration": fast_time,
            "fast_speedup": speedup,
            "sample_reference": similarity_ref,
        }
        summary.append(entry)

    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    for entry in summary:
        print(
            f"{entry['dataset']} n_jobs={entry['n_jobs']} "
            f"Δtrust={entry['trustworthiness_delta']:.4f} "
            f"overlap={entry['knn_overlap']:.3f} "
            f"corr={entry['pairwise_corr']:.4f} "
            f"speedup={entry['fast_speedup']:.2f}"
        )


if __name__ == "__main__":
    main()
