<p align="center">
  <h2 align="center">UMATO</h2>
  <h3 align="center">Uniform Manifold Approximation with Two-phase Optimization</h3>
</p>

---

> **Updated paper:** IEEE TVCG (2025, author version)  
> DOI: [10.1109/TVCG.2025.3602735](https://doi.org/10.1109/TVCG.2025.3602735)

UMATO is a dimensionality reduction (DR) technique designed to preserve both **local neighborhoods** and **global manifold relationships** in high-dimensional data. Existing DR methods often prioritize one side and can lead to misleading interpretations of manifold arrangement. UMATO addresses this with a two-phase optimization strategy and improves reliability for visual analytics.

## Key Contributions

- **Bridges local and global structures** in a single projection workflow.
- **Two-phase optimization**:
  1. Build a global skeletal layout using representative (hub) points.
  2. Project and optimize remaining points while preserving regional characteristics.
- **Improved stability** against initialization and subsampling variation.
- **Strong scalability** and competitive runtime on large datasets.

## System Requirements
- Python 3.9 or greater
- scikit-learn
- numpy
- scipy
- numba

## Installation

UMATO is available via pip.

```sh
pip install umato
```

## Quickstart

```python
import umato
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
emb = umato.UMATO(hub_num=50).fit_transform(X)
```

For detailed algorithm background, see the [API documentation](https://github.com/hyungkwonko/umato/wiki/API).

## API Reference

### Main Class

```python
from umato import UMATO

model = UMATO(
    n_neighbors=50,
    n_components=2,
    hub_num=300,
    metric="euclidean",
    global_n_epochs=100,
    local_n_epochs=50,
    global_learning_rate=0.0065,
    local_learning_rate=0.01,
    min_dist=0.1,
    spread=1.0,
    gamma=0.1,
    negative_sample_rate=5,
    init="pca",
    random_state=42,
    verbose=False,
)
```

### Key Parameters

- `n_neighbors` (int, default=50): neighborhood size used to build local structure.
- `hub_num` (int, default=300): number of representative hubs used in the global phase (`2 <= hub_num < n_samples`).
- `n_components` (int, default=2): output embedding dimensionality.
- `metric` (str, default=`"euclidean"`): distance metric for neighbor search (e.g., `"euclidean"`, `"cosine"`, `"precomputed"`).
- `init` (`"pca" | "random" | "spectral"` or ndarray, default=`"pca"`): initialization for hub layout.
- `global_n_epochs` / `local_n_epochs` (int): optimization epochs for each phase (defaults: 100 / 50).
- `global_learning_rate` / `local_learning_rate` (float): learning rates for global and local optimization.
- `min_dist`, `spread`: shape parameters controlling embedding compactness and spacing.
- `gamma` (float, default=0.1): repulsion strength in local optimization.
- `negative_sample_rate` (int, default=5): number of negative samples per positive edge.
- `random_state`: seed or `RandomState` for reproducibility.
- `verbose` (bool): print progress logs.

### Methods

- `fit(X)`: learn embedding from input `X`.
- `fit_transform(X) -> ndarray`: fit and return low-dimensional embedding.

### Attributes (after fitting)

- `embedding_`: final embedding of shape `(n_samples, n_components)`.
- `graph_`: fuzzy simplicial graph used in optimization.

### Example

```python
import umato
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
emb = umato.UMATO(
    n_neighbors=30,
    hub_num=50,
    init="pca",
    random_state=42,
).fit_transform(X)
```

## When to Use UMATO

UMATO is particularly useful when you need to:

- Inspect **cluster-level/global manifold arrangement** without giving up local neighborhood readability.
- Reduce interpretation risk from projections that over-emphasize either local compactness or global distance alone.
- Analyze high-dimensional datasets with a focus on **reliable visual analytics**.

## Findings

Detailed statistical data supporting UMATO’s accuracy and scalability are shown below.

#### Figure 1: Accuracy Analysis between Dimensionality Reduction Techniques
![Figure 1](images/figure1.png)  
Average scores of nine DR techniques in the accuracy analysis. UMATO substantially outperforms baselines in global metrics with a slight sacrifice in local metrics.

#### Figure 2: Local and Global Metric Rankings
<div align="center">
    <img src="images/figure2.png" width="60%">
</div>
Ranking by local/global quality metrics. UMATO shows the strongest global-structure performance among compared methods.

#### Figure 3: Scalability with Large Datasets
<div align="center">
    <img src="images/figure3.png" width="60%">
</div>
Runtime analysis for large datasets. UMATO outperforms most nonlinear baselines and shows strong practical scalability.

#### Figure 4: Projection Subset Analysis
![Figure 4](images/figure4.png)  
Projection subsets from the accuracy analysis. UMATO preserves global arrangement while remaining competitive on local structure.

#### Figure 5: Scalability with Small Datasets
<div align="center">
    <img src="images/figure5.png" width="60%">
</div>
Runtime analysis for small datasets. UMATO remains efficient while maintaining projection quality.

## Citation

### IEEE TVCG (2025) — Recommended

```bibtex
@article{jeon2025umato,
  title={UMATO: Bridging Local and Global Structures for Reliable Visual Analytics with Dimensionality Reduction},
  author={Jeon, Hyeon and Ko, Kwon and Lee, Soohyun and Hyun, Jake and Yang, Taehyun and Go, Gyehun and Jo, Jaemin and Seo, Jinwook},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  doi={10.1109/TVCG.2025.3602735}
}
```

### IEEE VIS (2022) — Original Conference Paper

```bibtex
@inproceedings{jeon2022vis,
  title={Uniform Manifold Approximation with Two-phase Optimization},
  author={Jeon, Hyeon and Ko, Hyung-Kwon and Lee, Soohyun and Jo, Jaemin and Seo, Jinwook},
  booktitle={2022 IEEE Visualization and Visual Analytics (VIS)},
  pages={80--84},
  year={2022},
  organization={IEEE}
}
```