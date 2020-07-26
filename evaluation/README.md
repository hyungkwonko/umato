# Run & Evaluate results both qualitatively and quantitatively

## Training models & Generating embedding result
We will generate embedding results for each algorithm for the comparison. The algorithms we will use are the following:
- t-SNE
- Topological Autoencoder
- UMAP
- UMATO (ours, TODO)

We can run each method separately, or all of them at once.
```python
# run all datasets
bash run-benchmark.sh

# run specific dataset (e.g., MNIST dataset)
bash run-benchmark.sh mnist
```

## Qualitative evaluation (TODO)
For the qualitative evaluation, we can compare the 2D visualization of each algorithm. We used the python library [dash](https://github.com/plotly/dash) for the visualization.

```python
# see visualization
python -m evaluation.visualization
```

### Embedding results of Fashion MNIST dataset for each algorithm:

|            t-SNE         |   Topological Autoencoder   |
:-------------------------:|:----------------------------:
![Fashion MNIST tsne](./images/fmnist/tsne.png)|![Fashion MNIST topoae](./images/fmnist/topoae.png)


|            UMAP             |           UMATO           |
:----------------------------:|:--------------------------:
![Fashion MNIST umap](./images/fmnist/umap.png)|![Fashion MNIST umato](./images/fmnist/umato.png)


## Quantitative evaluation (TODO)
Likewise, we can compare the embedding result quantitatively. We use measures such as RMSE, MRRE, Trustworthiness, continuity and KL divergence between density distributions for comparison. This will generate 4 * 5 table containing measures for each algorithm.


### Quantitative measures of Fashion MNIST dataset for each algorithm:

|                     |  PCA   | topoae | t-SNE  |  UMAP  |  UMATO (ours) |
| :-----------------: | :----: | :----: | :----: | :----: | :-----------: |
| RMSE                | 0.9316 | 0.9316 | 0.9281 | 0.9338 |               |
| MRRE                | 0.9316 | 0.9316 | 0.9281 | 0.9338 |               |
| TRUST               | 0.9272 | 0.9325 | 0.9206 | 0.9316 |               |
| Continuity          | 0.8187 | 0.8332 | 0.9360 | 0.9181 |               |
| KL-Div (sigma=1.)   | x.xxxx | x.xxxx | x.xxxx | x.xxxx |               |
| KL-Div (sigma=0.1)  | x.xxxx | x.xxxx | x.xxxx | x.xxxx |               |
| KL-Div (sigma=0.01) | x.xxxx | x.xxxx | x.xxxx | x.xxxx |               |

- RMSE: Lower is better
- MRRE: Higher is better
- Truestworthiness: Higher is better
- continuity: Higher is better
- KL divergence: Lower is better

```python
# see table result
python -m evaluation.comparison --algo=all --data=spheres --measure=all
```

## References
- Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. JMLR, 9(Nov), 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.
- Moor, M., Horn, M., Rieck, B., & Borgwardt, K. (2020). Topological autoencoders. ICML.