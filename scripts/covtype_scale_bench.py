#!/usr/bin/env python3
"""Benchmark deterministic vs. fast UMATO across growing covtype samples and plot runtime."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_covtype

from umato.umato_ import UMATO

BASE_DIR = Path("benchmarks")
BASE_DIR.mkdir(exist_ok=True)

SIZES = list(range(20_000, 200_001, 20_000))
MODES = ["deterministic", "fast"]
N_JOBS = 10
GLOBAL_EPOCHS = 11
LOCAL_EPOCHS = 11
RANDOM_STATE = 42


def run_single(size: int, mode: str, X: np.ndarray, perm: np.ndarray) -> float:
    subset = np.ascontiguousarray(X[perm[:size]])
    params = dict(
        n_neighbors=15,
        hub_num=50,
        global_n_epochs=GLOBAL_EPOCHS,
        local_n_epochs=LOCAL_EPOCHS,
        metric="euclidean",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        execution_mode=mode,
    )
    model = UMATO(**params)
    start = time.perf_counter()
    _ = model.fit_transform(subset)
    return time.perf_counter() - start


def main() -> None:
    X, _ = fetch_covtype(return_X_y=True)
    X = X.astype(np.float32, copy=False)
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(X.shape[0])

    # Warm-up to stabilize numba/JIT compilation prior to timing
    warm_subset = np.ascontiguousarray(X[perm[:2000]])
    UMATO(
        n_neighbors=15,
        hub_num=50,
        global_n_epochs=GLOBAL_EPOCHS,
        local_n_epochs=LOCAL_EPOCHS,
        metric="euclidean",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        execution_mode="deterministic",
    ).fit_transform(warm_subset)

    results = []
    for size in SIZES:
        for mode in MODES:
            duration = run_single(size, mode, X, perm)
            entry = {"size": size, "mode": mode, "duration": duration}
            results.append(entry)
            print(f"{mode:>12} size={size:6} duration={duration:.2f}s")

    csv_path = BASE_DIR / "covtype_scale.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["size", "mode", "duration"])
        writer.writeheader()
        writer.writerows(results)

    plt.figure(figsize=(6, 4))
    for mode in MODES:
        xs = [r["size"] for r in results if r["mode"] == mode]
        ys = [r["duration"] for r in results if r["mode"] == mode]
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xlabel("Sample Size")
    plt.ylabel("Duration (s)")
    plt.title("UMATO Runtime: deterministic vs fast (n_jobs=10)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "covtype_scale.png")
    print(f"Saved data to {csv_path} and plot to {BASE_DIR / 'covtype_scale.png'}")


if __name__ == "__main__":
    main()
