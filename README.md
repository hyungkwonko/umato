<p align="center">
  <h2 align="center">UMATO</h2>
	<h3 align="center">Uniform Manifold Approximation with Two-phase Optimization</h3>
</p>

---



Uniform Manifold Approximation with Two-phase Optimization (UMATO) is a dimensionality reduction technique, which can preserve the global as well as the local structure of high-dimensional data. Most existing dimensionality reduction algorithms focus on either of the two aspects, however, such insufficiency can lead to overlooking or misinterpreting important patterns in the data. For this aim, we propose a two-phase optimization: global optimization and local optimization. First, we obtain the global structure by selecting and optimizing the hub points.
Next, we initialize and optimize other points using the nearest neighbor graph. Our experiments with one synthetic and three real world datasets show that UMATO can outperform the baseline algorithms, such as PCA, [t-SNE](https://lvdmaaten.github.io/tsne/), [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html), [UMAP](https://github.com/lmcinnes/umap), [Topological Autoencoders](https://github.com/BorgwardtLab/topological-autoencoders) and [Anchor t-SNE](https://github.com/ZJULearning/AtSNE), in terms of global measures and qualitative projection results.

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

#### Parameters

```python
n_neighbors = 50
```
The size of the local neighborhood (defined by the number of nearby sample points) used for manifold approximation. Bigger values lead to a more comprehensive view of the manifold, whereas smaller values retain more local information. Generally, values should fall within the range of 2 to 100. It must be an integer greater than 1.
Same effect as it does in UMAP.

```python
min_dist = 0.1
```
 The minimum distance between embedded points. Same effect as it does in UMAP.


```python
n_components = 2
```
The dimensionality of the output embedding space. It must be a positive integer. This defaults to 2, but can reasonably be set to any integer value below the number of the original dataset dimensions.



```python
hub_num = 300
```
Number of hub points to use for the embedding. It must be a positive integer or -1 (None).


```python
metric = "euclidean"
```
The metric to use to compute distances in high dimensional space. If a string is passed it must match a valid predefined metric. If a general metric is required a function that takes two 1d arrays and returns a float can be provided. For performance purposes it is required that this be a numba jit’d function.

The default distance function is a Euclidean. The list of available options can be found in [the source code](https://github.com/hyungkwonko/umato/blob/master/src/umato/distances.py).

```python
global_n_epochs = None
```
The number of epochs for the global optimization phase. It must be a positive integer of at least 10. If not defiend, it will be set to 100.

```python
local_n_epochs = None 
```
The number of epochs for the local optimization phase. It must be a positive integer of at least 10. If not defined, it will be set to 50.


```python
global_learning_rate = 0.0065,
```
The learning rate for the global optimization phase. It must be positive.

```python
local_learning_rate = 0.01
```
The learning rate for the local optimization phase. It must be positive.

```python
spread = 1.0
```
Determines the scale at which embedded points will be spread out. Higher values will lead to more separation between points.
`min_dist` must be less than or equal to `spread`.

```python
low_memory = False
```
Whether to use a lower memory, but more computationally expensive approach to construct k-nearest neighbor graph

```python
set_op_mix_ratio
```
Interpolation parameter for combining the global and local structures in the fuzzy simplicial set. It must be between ```0.0``` and ```1.0```.
A value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.

```python
local_connectivity = 1.0
```
The local connectivity required – i.e. the number of nearest neighbors that should be assumed to be connected at a local level. The higher this value the more connected the manifold becomes locally.

```python
gamma = 0.1
```
The gamma parameter used in local optimization for adjusting the balance between attractive and repulsive forces. It must be non-negative.

```python
negative_sample_rate = 5
```
The number of negative samples to select per positive sample in the optimization process. Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.


```python
random_state = None
```
The seed of the pseudo random number generator to use when shuffling the data. 

```python
angular_rp_forest = False
```

A boolean flag that indicates whether to use angular random projection forest for approximate nearest neighbor search. It is set to True if the ```self.metric``` is in the set ```{"cosine", "correlation", "dice", "jaccard", "ll_dirichlet", "hellinger"}```.


These parameters, along with their conditions and constraints, control various aspects of the embedding process, including the distance metric, optimization settings, and the structure of the resulting low-dimensional space.

Whether to utilize an angular random projection forest for initializing the approximate nearest neighbor search. This approach can offer improved speed, but it is primarily beneficial for metrics employing an angular-style distance, such as cosine, correlation, and others. For these metrics, angular forests will be automatically selected.

```python
init = "pca"
```

The initialization method to use for the embedding. It must be a string or a numpy array. If a string is passed it must match one of the following: `init`, `random`, `spectral`.


```python
verbose = False
```
Whether to print information about the optimization process.






### Function `fit`

```python 
def fit(self, X):
```

This ```fit``` function embeds the input data X into a lower-dimensional space. It handles optional arguments, validates parameters, checks the data sparsity, and builds the nearest neighbor graph structure. It also computes global and local optimization, initializes the embedding using the original hub information, and embeds outliers.

After validating the input data and setting default values for optional arguments, the function checks if the metric is supported by ```PyNNDescent``` and computes the nearest neighbors accordingly. It then builds the k-nearest neighbor graph structure, runs global and local optimization, and embeds the remaining data points. Finally, it embeds outliers and returns the **fitted model**.


#### Parameters

``` python
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
```
           
 If the metric is 'precomputed' X must be a square distance matrix. Otherwise it contains a sample per row. If the method is 'exact', X may be a sparse matrix of type 'csr', 'csc' or 'coo'.


### Function `fit_transform`

```python
def fit_transform(self, X):
```

Fit X into an embedded space and return that transformed output.

#### Parameters

```python
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
```
If the metric is ‘precomputed’ X must be a square distance matrix. Otherwise it contains a sample per row.

#### Returns

 ```X_new : array, shape (n_samples, n_components)```
Embedding of the training data in low-dimensional space.



## Evaluation

### Training models & Generating embedding result
We will generate embedding results for each algorithm for the comparison. The algorithms we will use are the following:
- PCA
- [t-SNE](https://lvdmaaten.github.io/tsne/)
- [UMAP](https://github.com/lmcinnes/umap)
- [Topological Autoencoder](https://github.com/BorgwardtLab/topological-autoencoders)
- [Anchor t-SNE](https://github.com/ZJULearning/AtSNE)
- UMATO (ours)

We can run each method separately, or all of them at once.
```python
# run all datasets
bash run-benchmark.sh

# run specific dataset (e.g., MNIST dataset)
bash run-benchmark.sh mnist
```
This will cover PCA, t-SNE, UMAP and Topological Autoencoders. To run Anchor t-SNE, you need CUDA and GPU. Please refer to [here](https://github.com/ZJULearning/AtSNE) for specification.




## Quantitative evaluation
Likewise, we compared the embedding result quantitatively. We use measures such as Distance to a measure and KL divergence between density distributions for comparison.

To print the quantitative result:
```python
# print table result
python -m evaluation.comparison --algo=all --data=spheres --measure=all
```

### Result for the Spheres dataset

|                     |  PCA   | Isomap | t-SNE  |  UMAP  | TopoAE | At-SNE |  UMATO (ours) |
| :-----------------: | :----: | :----: | :----: | :----: | :----: | :----: | :-----------: |
| DTM                 | 0.9950 | 0.7784 | 0.9116 | 0.9209 | __0.6619__ | 0.9448 | __0.3849__    |
| KL-Div (sigma=0.01) | 0.7568 | 0.4492 | 0.6070 | 0.6100 | __0.1865__ | 0.6584 | __0.1569__    |
| KL-Div (sigma=0.1)  | 0.6525 | 0.4267 | 0.5365 | 0.5383 | __0.3007__ | 0.5712 | __0.1333__    |
| KL-Div (sigma=1.)   | 0.0153 | 0.0095 | 0.0128 | 0.0134 | __0.0057__ | 0.0138 | __0.0008__    |
| Cont                | 0.7983 | __0.9041__ | __0.8903__ | 0.8760 | 0.8317 | 0.8721 | 0.7884    |
| Trust               | 0.6088 | 0.6266 | __0.7073__ | 0.6499 | 0.6339 | 0.6433 | __0.6558__    |
| MRRE_X              | 0.7985 | __0.9039__ | __0.9032__ | 0.8805 | 0.8317 | 0.8768 | 0.7887    |
| MRRE_Z              | 0.6078 | 0.6268 | __0.7261__ | 0.6494 | 0.6326 | 0.6424 | __0.6557__    |

- DTM & KL divergence: Lower is better
- The winnder and runner-up is in bold.


## References
- Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. JMLR, 9(Nov), 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.
- Moor, M., Horn, M., Rieck, B., & Borgwardt, K. (2020). Topological autoencoders. ICML.
- Fu, C., Zhang, Y., Cai, D., & Ren, X. (2019, July). AtSNE: Efficient and Robust Visualization on GPU through Hierarchical Optimization. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 176-186).

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
