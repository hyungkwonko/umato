import numpy as np
import pytest
from typing import Optional

from umato.umato_ import UMATO


def _prepare_features(data: np.ndarray, max_rows: int = 3000) -> np.ndarray:
    x = np.asarray(data)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.astype(np.float32, copy=False)

    if x.shape[0] > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(x.shape[0], size=max_rows, replace=False)
        x = x[idx]

    return np.ascontiguousarray(x)


@pytest.mark.slow
@pytest.mark.datasets
@pytest.mark.timeout(180)
def test_dataset_smoke_all(
    dataset_entry: Optional[dict[str, object]],
    dataset_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    if dataset_entry is None:
        pytest.skip("No dataset entry available")

    dataset_name = str(dataset_entry["name"])
    x_raw, _ = dataset_arrays
    x = _prepare_features(x_raw)

    if x.shape[0] < 3:
        pytest.skip(f"Dataset {dataset_name} has fewer than 3 samples")

    params = {
        "n_neighbors": min(15, x.shape[0] - 1),
        "hub_num": max(2, min(50, x.shape[0] - 1)),
        "metric": "euclidean",
        "init": "pca",
        "global_n_epochs": 11,
        "local_n_epochs": 11,
        "random_state": 42,
    }

    try:
        emb = UMATO(**params).fit_transform(x)
    except Exception as exc:
        pytest.fail(f"dataset={dataset_name} params={params} failed: {exc}")

    assert emb.shape == (x.shape[0], 2), f"dataset={dataset_name} params={params}"
    assert np.isfinite(emb).all(), f"dataset={dataset_name} params={params}"
