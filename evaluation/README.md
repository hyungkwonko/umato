# Running & comparing embedding results both quantitatively & quanlitatively

## Training models & Generating embedding result
We will generate embedding results for each algorithm for the comparison. The algorithms we will use are the following:
- Topo-ae
- t-SNE
- UMAP
- UMATO (ours, TODO)

We can run each method separately, or all of them at once.
```python
# run all datasets
sh run.sh

# run specific dataset (e.g., MNIST dataset)
sh run.sh mnist
```

## Qualitative evaluation (TODO)
For the qualitative evaluation, we can compare the 2D visualization of each algorithm. We used the python library [dash](https://github.com/plotly/dash) for the visualization.

```python
# run visualization
python visualization.py
```

## Quantitative evaluation (TODO)
Likewise, we can compare the embedding result quantitatively. We use measures such as X1, X2, ... for comparison. This will generate 4 * 5 table containing measures for each algorithm.

```python
# run comparison
python comparison.py
```