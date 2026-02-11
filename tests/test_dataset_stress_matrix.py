import numpy as np
import pytest
from typing import Optional

from umato.umato_ import UMATO


def _prepare_features(data: np.ndarray, max_rows: int = 4000) -> np.ndarray:
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
@pytest.mark.timeout(240)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("init", ["pca", "random"])
@pytest.mark.parametrize("requested_n_neighbors", [5, 15, 50])
@pytest.mark.parametrize("requested_hub_num", [10, 50, 100])
def test_dataset_stress_matrix(
    representative_dataset_entry: Optional[dict[str, object]],
    metric: str,
    init: str,
    requested_n_neighbors: int,
    requested_hub_num: int,
) -> None:
    if representative_dataset_entry is None:
        pytest.skip("No representative dataset entry available")

    dataset_name = str(representative_dataset_entry["name"])
    data_path = representative_dataset_entry["data_path"]
    x_raw = np.load(data_path, allow_pickle=False)
    x = _prepare_features(x_raw)

    if x.shape[0] < 3:
        pytest.skip(f"Dataset {dataset_name} has fewer than 3 samples")

    n_neighbors = min(requested_n_neighbors, x.shape[0] - 1)
    hub_num = max(2, min(requested_hub_num, x.shape[0] - 1))

    if n_neighbors < 2 or hub_num < 2:
        pytest.skip(
            f"Invalid effective params for dataset={dataset_name}: "
            f"n_neighbors={n_neighbors}, hub_num={hub_num}"
        )

    params = {
        "n_neighbors": n_neighbors,
        "hub_num": hub_num,
        "metric": metric,
        "init": init,
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
