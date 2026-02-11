#!/usr/bin/env python3
"""Benchmark UMATO on large labeled datasets with deterministic and fast execution modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_rcv1
from sklearn.preprocessing import OneHotEncoder

from umato.umato_ import UMATO

SAMPLE_SIZE = 20_000
SIMILARITY_REFERENCE_SIZE = 300
DEFAULT_N_JOBS = [1, 2, 4, 8, 10]
DEFAULT_EXECUTION_MODES = ["deterministic", "fast"]


def _covtype_loader() -> np.ndarray:
    X, _ = fetch_covtype(return_X_y=True)
    return np.ascontiguousarray(X.astype(np.float32, copy=False))


def _rcv1_loader() -> sp.csr_matrix:
    X, _ = fetch_rcv1(return_X_y=True)
    return X.tocsr().astype(np.float32)


def _kddcup99_loader() -> sp.csr_matrix:
    X, _ = fetch_kddcup99(
        return_X_y=True,
        subset=None,
        percent10=True,
        shuffle=True,
        random_state=42,
    )
    X = np.asarray(X)
    n_features = X.shape[1]
    categorical_cols = [1, 2, 3]
    numeric_cols = [c for c in range(n_features) if c not in categorical_cols]

    transformer = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ],
        sparse_threshold=1.0,
    )
    Xt = transformer.fit_transform(X)
    if not sp.isspmatrix_csr(Xt):
        Xt = Xt.tocsr()
    return Xt.astype(np.float32)


DATASETS = [
    {
        "name": "covtype",
        "loader": _covtype_loader,
        "metric": "euclidean",
    },
    {
        "name": "rcv1",
        "loader": _rcv1_loader,
        "metric": "cosine",
    },
    {
        "name": "kddcup99",
        "loader": _kddcup99_loader,
        "metric": "euclidean",
    },
]


def _limit_rows(X, indices: np.ndarray) -> np.ndarray | sp.csr_matrix:
    if sp.issparse(X):
        return X[indices]
    return X[indices]


def _prepare_similarity_reference(
    dataset_name: str,
    X: np.ndarray | sp.csr_matrix,
    rng: np.random.Generator,
    similarity_dir: Path,
) -> Path:
    target = similarity_dir / f"{dataset_name}-sample.npz"
    if target.exists():
        return target

    n_samples = X.shape[0]
    size = min(SIMILARITY_REFERENCE_SIZE, n_samples)
    indices = rng.choice(n_samples, size=size, replace=False)
    indices.sort()

    subset = _limit_rows(X, indices)
    if sp.issparse(subset):
        dense = subset.toarray()
    else:
        dense = np.ascontiguousarray(subset)

    np.savez_compressed(
        target,
        indices=indices,
        data=np.asarray(dense, dtype=np.float32),
    )
    return target


def run_benchmarks(
    results_path: Path,
    embeddings_dir: Path,
    similarity_dir: Path,
    sample_size: int,
    n_jobs_list: list[int],
    execution_modes: list[str],
    random_state: int,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    similarity_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)
    dataset_cache: dict[str, np.ndarray | sp.csr_matrix] = {}

    with results_path.open("a", encoding="utf-8") as out_file:
        for dataset in DATASETS:
            name = dataset["name"]
            if name not in dataset_cache:
                dataset_cache[name] = dataset["loader"]()
            full_data = dataset_cache[name]
            metric = dataset["metric"]

            dataset_rng = np.random.default_rng(random_state + hash(name) % 1_000)

            scenarios = [("full", None)]
            scenarios.append(("sample", sample_size))

            similarity_reference = None
            sample_indices = None

            for scenario, limit in scenarios:
                if limit is None:
                    subset = full_data
                    n_samples = subset.shape[0]
                    similarity_reference_path = None
                else:
                    if similarity_reference is None:
                        similarity_reference = _prepare_similarity_reference(
                            name, full_data, dataset_rng, similarity_dir
                        )
                    with np.load(similarity_reference) as buf:
                        indices = buf["indices"]
                    subset = _limit_rows(full_data, indices)
                    n_samples = subset.shape[0]
                    similarity_reference_path = str(similarity_reference)

                if n_samples < 3:
                    continue

                n_neighbors = min(15, n_samples - 1)
                hub_num = max(2, min(50, n_samples - 1))

                for mode in execution_modes:
                    for n_jobs in n_jobs_list:
                        params = {
                            "n_neighbors": n_neighbors,
                            "hub_num": hub_num,
                            "metric": metric,
                            "init": "pca",
                            "global_n_epochs": 11,
                            "local_n_epochs": 11,
                            "execution_mode": mode,
                            "n_jobs": n_jobs,
                            "random_state": random_state,
                        }

                        run_id = f"{name}-{scenario}-{mode}-njobs{n_jobs}"
                        start = time.perf_counter()
                        try:
                            model = UMATO(**params)
                            emb = model.fit_transform(subset)
                            success = True
                            error = None
                            embedding_path = None
                            if scenario == "sample":
                                emb_path = embeddings_dir / f"{run_id}.npy"
                                np.save(emb_path, emb.astype(np.float32))
                                embedding_path = str(emb_path)
                        except Exception as exc:
                            success = False
                            error = str(exc)
                            embedding_path = None
                        duration = time.perf_counter() - start

                        entry = {
                            "run_id": run_id,
                            "dataset": name,
                            "scenario": scenario,
                            "metric": metric,
                            "n_jobs": n_jobs,
                            "execution_mode": mode,
                            "n_samples": n_samples,
                            "n_features": subset.shape[1],
                            "params": params,
                            "duration_sec": duration,
                            "success": success,
                            "error": error,
                            "embedding_path": embedding_path,
                            "similarity_reference": similarity_reference_path,
                        }

                        out_file.write(json.dumps(entry) + "\n")
                        out_file.flush()
                        status = "ok" if success else "fail"
                        print(
                            f"[{status.upper()}] {run_id} "
                            f"mode={mode} n_jobs={n_jobs} "
                            f"duration={duration:.2f}s"
                        )


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("benchmarks/results.jsonl"),
        help="JSONL path where benchmark entries are appended",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("benchmarks/embeddings"),
        help="Directory to persist sample embeddings",
    )
    parser.add_argument(
        "--similarity-dir",
        type=Path,
        default=Path("benchmarks/similarity"),
        help="Directory that stores sample feature references",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Maximum number of rows when running the sample scenario",
    )
    parser.add_argument(
        "--n-jobs",
        type=str,
        default="1,2,4,8,10",
        help="Comma-separated list of n_jobs values",
    )
    parser.add_argument(
        "--execution-modes",
        type=str,
        default="deterministic,fast",
        help="Comma-separated execution modes to run",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )
    args = parser.parse_args()

    n_jobs = parse_int_list(args.n_jobs) or DEFAULT_N_JOBS
    modes = [mode.strip() for mode in args.execution_modes.split(",") if mode.strip()]
    if not modes:
        modes = DEFAULT_EXECUTION_MODES

    run_benchmarks(
        results_path=args.results_file,
        embeddings_dir=args.embeddings_dir,
        similarity_dir=args.similarity_dir,
        sample_size=args.sample_size,
        n_jobs_list=n_jobs,
        execution_modes=modes,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
