import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import load_iris

from umato.umato_ import UMATO


@pytest.fixture(scope="module")
def iris_sparse() -> sp.csr_matrix:
    x, _ = load_iris(return_X_y=True)
    return sp.csr_matrix(x.astype(np.float32, copy=False))


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_sparse_input_smoke(iris_sparse: sp.csr_matrix, metric: str) -> None:
    emb = UMATO(
        metric=metric,
        n_neighbors=15,
        hub_num=20,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    ).fit_transform(iris_sparse)

    assert emb.shape == (iris_sparse.shape[0], 2)
    assert np.isfinite(emb).all()


def test_large_sparse_native_path_runs() -> None:
    rng = np.random.RandomState(42)
    large_sparse = sp.random(
        80,
        1_000_000,
        density=5e-6,
        format="csr",
        random_state=rng,
        dtype=np.float32,
    )

    model = UMATO(
        metric="cosine",
        n_neighbors=5,
        hub_num=10,
        global_n_epochs=11,
        local_n_epochs=11,
        random_state=42,
    )
    emb = model.fit_transform(large_sparse)

    assert emb.shape == (80, 2)
    assert np.isfinite(emb).all()
