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


def test_sparse_input_memory_guard() -> None:
    large_sparse = sp.csr_matrix((100_000, 100_000), dtype=np.float32)
    with pytest.raises(ValueError, match="too large to densify safely"):
        UMATO(n_neighbors=2, hub_num=2, random_state=42).fit(large_sparse)
