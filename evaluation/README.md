# Run & Evaluate results both quantitatively and quanlitatively

## Training models & Generating embedding result
We will generate embedding results for each algorithm for the comparison. The algorithms we will use are the following:
- Topo-ae
- t-SNE
- UMAP
- UMATO (ours, TODO)

We can run each method separately, or all of them at once.
```python
# run all datasets
bash run.sh

# run specific dataset (e.g., MNIST dataset)
bash run.sh mnist
```

## Qualitative evaluation (TODO)
For the qualitative evaluation, we can compare the 2D visualization of each algorithm. We used the python library [dash](https://github.com/plotly/dash) for the visualization.

```python
# see visualization
python visualization.py
```

## Quantitative evaluation (TODO)
Likewise, we can compare the embedding result quantitatively. We use measures such as RMSE, MRRE, Trustworthiness, continuity and KL-divergence for comparison. This will generate 4 * 5 table containing measures for each algorithm.


```python
# see table result
python comparison.py
```

## References
- Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. JMLR, 9(Nov), 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.
- Moor, M., Horn, M., Rieck, B., & Borgwardt, K. (2020). Topological autoencoders. ICML.