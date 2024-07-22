<p align="center">
  <h2 align="center">UMATO</h2>
	<h3 align="center">Uniform Manifold Approximation with Two-phase Optimization</h3>
</p>

---



Uniform Manifold Approximation with Two-phase Optimization (UMATO) is a dimensionality reduction technique, which can preserve the global as well as the local structure of high-dimensional data. Most existing dimensionality reduction algorithms focus on either of the two aspects, however, such insufficiency can lead to overlooking or misinterpreting important global patterns in the data. Moreover, the existing algorithms suffer from instability. 
To address these issues, UMATO proposes a two-phase optimization: global optimization and local optimization. First, we obtain the global structure by selecting and optimizing the hub points.
Next, we initialize and optimize other points using the nearest neighbor graph. Our experiments with one synthetic and three real world datasets show that UMATO can outperform the baseline algorithms, such as PCA, [t-SNE](https://lvdmaaten.github.io/tsne/), [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html), [UMAP](https://github.com/lmcinnes/umap), [LAMP](https://github.com/lgnonato/LAMP) and [PacMAP](https://github.com/YingfanWang/PaCMAP), in terms of accuracy, stability, and scalability.

## System Requirements
- Python 3.9 or greater
- scikit-learn
- numpy
- scipy
- numba
- pandas (to read csv files)

## Installation 

UMATO is available via pip.

```sh
pip install umato
```

```python
import umato
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
emb = umato.UMATO(hub_num=50).fit_transform(X)
```

## API

### Default Class: UMATO

Umato has only one class ```UMATO```. Note that UMATO shares a bunch of parameters with UMAP. For more details, please refer to [UMAP API](https://umap-learn.readthedocs.io/en/latest/api.html).



```python
class UMATO(BaseEstimator):
    def __init__(
        self,
        n_neighbors=50,
	min_dist=0.1,
        n_components=2,
        hub_num=300,
        metric="euclidean",
        global_n_epochs=None,
        local_n_epochs=None,
        global_learning_rate=0.0065,
        local_learning_rate=0.01,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        gamma=0.1,
        negative_sample_rate=5,
        random_state=None,
        angular_rp_forest=False,
        init="pca",
	verbose=False
    ):

```

### Function `fit`

```python 
def fit(self, X):
```

This ```fit``` function embeds the input data X into a lower-dimensional space. It handles optional arguments, validates parameters, checks the data sparsity, and builds the nearest neighbor graph structure. It also computes global and local optimization, initializes the embedding using the original hub information, and embeds outliers.

After validating the input data and setting default values for optional arguments, the function checks if the metric is supported by ```PyNNDescent``` and computes the nearest neighbors accordingly. It then builds the k-nearest neighbor graph structure, runs global and local optimization, and embeds the remaining data points. Finally, it embeds outliers and returns the **fitted model**.

For detailed parameter usage, check the API listed under Wiki.


## Citation

UMATO can be cited as follows:

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

Jeon, H., Ko, H. K., Lee, S., Jo, J., & Seo, J. (2022, October). Uniform Manifold Approximation with Two-phase Optimization. In 2022 IEEE Visualization and Visual Analytics (VIS) (pp. 80-84). IEEE.
