from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

DEFAULT_DATASETS_ROOT = Path("/Users/hj/Desktop/umato/external/labeled-datasets")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--datasets-root",
        action="store",
        default=None,
        help=(
            "Root directory of labeled-datasets clone. "
            "Defaults to $UMATO_DATASETS_ROOT or "
            "/Users/hj/Desktop/umato/external/labeled-datasets"
        ),
    )


def _resolve_datasets_root(config: pytest.Config) -> Path:
    cli_value = config.getoption("--datasets-root")
    env_value = os.getenv("UMATO_DATASETS_ROOT")
    root_value = cli_value or env_value or str(DEFAULT_DATASETS_ROOT)
    return Path(root_value).expanduser().resolve()


def _read_data_shape(path: Path) -> tuple[int, ...]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(array.shape)
    del array
    return shape


def discover_dataset_entries(datasets_root: Path) -> list[dict[str, object]]:
    npy_root = datasets_root / "npy"
    if not npy_root.exists():
        return []

    entries: list[dict[str, object]] = []
    for dataset_dir in sorted(p for p in npy_root.iterdir() if p.is_dir()):
        data_path = dataset_dir / "data.npy"
        label_path = dataset_dir / "label.npy"
        if not data_path.exists() or not label_path.exists():
            continue

        try:
            shape = _read_data_shape(data_path)
        except Exception:
            continue

        if len(shape) == 0:
            continue

        n_samples = int(shape[0])
        n_features = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        entries.append(
            {
                "name": dataset_dir.name,
                "dataset_dir": dataset_dir,
                "data_path": data_path,
                "label_path": label_path,
                "shape": shape,
                "n_samples": n_samples,
                "n_features": n_features,
            }
        )

    return entries


def _get_cached_entries(config: pytest.Config) -> list[dict[str, object]]:
    cache_attr = "_umato_dataset_entries_cache"
    cached = getattr(config, cache_attr, None)
    if cached is None:
        root = _resolve_datasets_root(config)
        cached = discover_dataset_entries(root)
        setattr(config, cache_attr, cached)
    return cached


def _select_representative_entries(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(entries) <= 3:
        return entries

    sizes = np.array([int(e["n_samples"]) for e in entries], dtype=np.float64)
    selected: dict[str, dict[str, object]] = {}

    for q in (0.1, 0.5, 0.9):
        target = float(np.quantile(sizes, q))
        idx = int(np.argmin(np.abs(sizes - target)))
        entry = entries[idx]
        selected[str(entry["name"])] = entry

    if len(selected) < 3:
        for entry in sorted(entries, key=lambda x: int(x["n_samples"])):
            selected.setdefault(str(entry["name"]), entry)
            if len(selected) >= 3:
                break

    return sorted(selected.values(), key=lambda x: int(x["n_samples"]))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dataset_entry" in metafunc.fixturenames:
        entries = _get_cached_entries(metafunc.config)
        if not entries:
            metafunc.parametrize(
                "dataset_entry",
                [
                    pytest.param(
                        None,
                        marks=pytest.mark.skip(
                            reason="No datasets found. Run scripts/prepare_labeled_datasets.sh"
                        ),
                        id="datasets-missing",
                    )
                ],
            )
        else:
            metafunc.parametrize(
                "dataset_entry",
                [pytest.param(entry, id=str(entry["name"])) for entry in entries],
            )

    if "representative_dataset_entry" in metafunc.fixturenames:
        entries = _get_cached_entries(metafunc.config)
        representatives = _select_representative_entries(entries)
        if not representatives:
            metafunc.parametrize(
                "representative_dataset_entry",
                [
                    pytest.param(
                        None,
                        marks=pytest.mark.skip(
                            reason="No datasets found. Run scripts/prepare_labeled_datasets.sh"
                        ),
                        id="datasets-missing",
                    )
                ],
            )
        else:
            metafunc.parametrize(
                "representative_dataset_entry",
                [
                    pytest.param(entry, id=f"rep-{entry['name']}")
                    for entry in representatives
                ],
            )


@pytest.fixture(scope="session")
def datasets_root(request: pytest.FixtureRequest) -> Path:
    return _resolve_datasets_root(request.config)


@pytest.fixture(scope="session")
def all_dataset_entries(request: pytest.FixtureRequest) -> list[dict[str, object]]:
    entries = _get_cached_entries(request.config)
    if not entries:
        pytest.skip("No datasets found. Run scripts/prepare_labeled_datasets.sh")
    return entries


@pytest.fixture
def dataset_arrays(dataset_entry: Optional[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    if dataset_entry is None:
        pytest.skip("No dataset entry available")

    data = np.load(Path(dataset_entry["data_path"]), allow_pickle=False)
    labels = np.load(Path(dataset_entry["label_path"]), allow_pickle=False)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    labels = np.ravel(labels)
    return data, labels
