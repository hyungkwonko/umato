import numpy as np
import pytest
from sklearn.datasets import load_iris

from umato.umato_ import UMATO


@pytest.fixture
def iris_data() -> np.ndarray:
    X, _ = load_iris(return_X_y=True)
    return X.astype(np.float32, copy=False)


def test_execution_mode_validation(iris_data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="execution_mode must be 'deterministic' or 'fast'"):
        UMATO(n_neighbors=5, hub_num=10, execution_mode="turbo").fit(iris_data)


def test_n_jobs_validation(iris_data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="n_jobs must be None, -1, or a positive integer"):
        UMATO(n_neighbors=5, hub_num=10, n_jobs=0).fit(iris_data)


@pytest.mark.parametrize("mode", ["deterministic", "fast"])
def test_execution_modes_run(iris_data: np.ndarray, mode: str) -> None:
    emb = UMATO(
        n_neighbors=5,
        hub_num=10,
        global_n_epochs=11,
        local_n_epochs=11,
        execution_mode=mode,
        n_jobs=2,
        random_state=42,
    ).fit_transform(iris_data)
    assert emb.shape == (iris_data.shape[0], 2)
