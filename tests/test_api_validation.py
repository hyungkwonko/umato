import numpy as np
import pytest
from sklearn.datasets import load_iris

from umato.umato_ import UMATO


@pytest.fixture(scope="module")
def iris_data() -> np.ndarray:
    x, _ = load_iris(return_X_y=True)
    return x.astype(np.float32, copy=False)


def test_hub_num_type_validation(iris_data: np.ndarray) -> None:
    for bad_value in (None, "10", 3.5):
        with pytest.raises(ValueError, match="hub_num must be an integer"):
            UMATO(hub_num=bad_value, n_neighbors=15, random_state=42).fit(iris_data)


def test_hub_num_range_validation(iris_data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="hub_num must be at least 2"):
        UMATO(hub_num=1, n_neighbors=15, random_state=42).fit(iris_data)

    with pytest.raises(ValueError, match="hub_num must be less than the number of data points"):
        UMATO(hub_num=iris_data.shape[0], n_neighbors=15, random_state=42).fit(iris_data)


@pytest.mark.parametrize("negative_sample_rate", [0, -1])
def test_negative_sample_rate_must_be_positive(
    iris_data: np.ndarray, negative_sample_rate: float
) -> None:
    with pytest.raises(ValueError, match="negative_sample_rate must be greater than 0"):
        UMATO(
            hub_num=20,
            n_neighbors=15,
            negative_sample_rate=negative_sample_rate,
            random_state=42,
        ).fit(iris_data)


def test_init_shape_validation(iris_data: np.ndarray) -> None:
    bad_rows = np.random.RandomState(0).normal(size=(iris_data.shape[0] - 1, 2)).astype(np.float32)
    bad_cols = np.random.RandomState(1).normal(size=(iris_data.shape[0], 1)).astype(np.float32)

    with pytest.raises(ValueError, match="init array first dimension must match n_samples"):
        UMATO(hub_num=20, n_neighbors=15, init=bad_rows, n_components=2, random_state=42).fit(iris_data)

    with pytest.raises(ValueError, match="init array second dimension must match n_components"):
        UMATO(hub_num=20, n_neighbors=15, init=bad_cols, n_components=2, random_state=42).fit(iris_data)


@pytest.mark.parametrize("n_neighbors", ["15", 1])
def test_n_neighbors_boundary_validation(iris_data: np.ndarray, n_neighbors) -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        UMATO(hub_num=20, n_neighbors=n_neighbors, random_state=42).fit(iris_data)
