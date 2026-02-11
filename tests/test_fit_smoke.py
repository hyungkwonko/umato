import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances

from umato.umato_ import UMATO


@pytest.fixture(scope="module")
def iris_data() -> np.ndarray:
    x, _ = load_iris(return_X_y=True)
    return x.astype(np.float32, copy=False)


def test_dense_fit_smoke(iris_data: np.ndarray) -> None:
    emb = UMATO(
        n_neighbors=15,
        hub_num=20,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    ).fit_transform(iris_data)

    assert emb.shape == (iris_data.shape[0], 2)
    assert np.isfinite(emb).all()


def test_precomputed_metric_smoke(iris_data: np.ndarray) -> None:
    dmat = pairwise_distances(iris_data, metric="euclidean").astype(np.float32)
    emb = UMATO(
        metric="precomputed",
        n_neighbors=15,
        hub_num=20,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    ).fit_transform(dmat)

    assert emb.shape == (iris_data.shape[0], 2)
    assert np.isfinite(emb).all()


def test_small_dataset_neighbor_truncation(iris_data: np.ndarray) -> None:
    x_small = iris_data[:10]
    model = UMATO(
        n_neighbors=50,
        hub_num=3,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    )

    with pytest.warns(UserWarning, match="truncating"):
        emb = model.fit_transform(x_small)

    assert emb.shape == (x_small.shape[0], 2)
    assert model._n_neighbors == x_small.shape[0] - 1
    assert model._knn_indices.shape[1] == x_small.shape[0] - 1


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_n_components_smoke(iris_data: np.ndarray, n_components: int) -> None:
    emb = UMATO(
        n_components=n_components,
        n_neighbors=15,
        hub_num=20,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    ).fit_transform(iris_data)

    assert emb.shape == (iris_data.shape[0], n_components)
    assert np.isfinite(emb).all()


def test_duplicate_points_knn_coverage_regression() -> None:
    rng = np.random.RandomState(0)
    x_dup = rng.randint(0, 3, size=(100, 2)).astype(np.float32)

    emb = UMATO(
        n_neighbors=5,
        hub_num=10,
        metric="euclidean",
        init="pca",
        negative_sample_rate=1,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    ).fit_transform(x_dup)

    assert emb.shape == (x_dup.shape[0], 2)
    assert np.isfinite(emb).all()
