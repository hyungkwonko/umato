# Uniform Manifold Approximation with Two-phase Optimization

### Notice

*Appreciate all the interests in UMATO at VIS 2022! We'll soon work on cleaning the codes and resolving the bugs (within Early 2023). Thank you!*


-----


Uniform Manifold Approximation with Two-phase Optimization (UMATO) is a dimensionality reduction technique, which can preserve the global as well as the local structure of high-dimensional data. Most existing dimensionality reduction algorithms focus on either of the two aspects, however, such insufficiency can lead to overlooking or misinterpreting important patterns in the data. For this aim, we propose a two-phase optimization: global optimization and local optimization. First, we obtain the global structure by selecting and optimizing the hub points.
Next, we initialize and optimize other points using the nearest neighbor graph. Our experiments with one synthetic and three real world datasets show that UMATO can outperform the baseline algorithms, such as PCA, [t-SNE](https://lvdmaaten.github.io/tsne/), [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html), [UMAP](https://github.com/lmcinnes/umap), [Topological Autoencoders](https://github.com/BorgwardtLab/topological-autoencoders) and [Anchor t-SNE](https://github.com/ZJULearning/AtSNE), in terms of global measures and qualitative projection results.

### System Requirements
- Python 3.6 or greater
- scikit-learn
- numpy
- scipy
- numba
- pandas (to read csv files)

### Installation 

UMATO is available via pip.

```sh
pip install umato
```

```python
import umato
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
emb = umato.UMATO(hub_num=20).fit_transform(X)
```

# Evaluation

## Training models & Generating embedding result
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
