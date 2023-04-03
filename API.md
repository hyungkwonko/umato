# UMATO API Markdown

Uniform Manifold Approximation with Two-phase Optimization

```umato_.py``` code specifications

[Source Code](https://github.com/hyungkwonko/umato/blob/master/src/umato/umato_.py#L560)

#

# Default UMATO Class

Umato has only one class ```UMATO```


```python
class UMATO(BaseEstimator):
    def __init__(
        self,
        n_neighbors=50,
        n_components=2,
        hub_num=300,
        metric="euclidean",
        global_n_epochs=None,
        local_n_epochs=None,
        global_learning_rate=0.0065,
        local_learning_rate=0.01,
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        gamma=0.1,
        negative_sample_rate=5,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        init="pca",
        ll=None,
    ):

```

---

## Class Parameters

```python
metric
```
The metric to use to compute distances in high dimensional space. If a string is passed it must match a valid predefined metric. If a general metric is required a function that takes two 1d arrays and returns a float can be provided. For performance purposes it is required that this be a numba jit’d function.

If it is in the set ```{"cosine", "correlation", "dice", "jaccard", "ll_dirichlet", "hellinger"}```, ```self.angular_rp_forest``` will be set to True. This parameter must be a string or a callable function. For sparse data, only specific metrics are supported.

#

```python
set_op_mix_ratio
```
Interpolation parameter for combining the global and local structures in the fuzzy simplicial set. It must be between ```0.0``` and ```1.0```.

#
```
gamma
```
The gamma parameter used in local optimization for adjusting the balance between attractive and repulsive forces. It must be non-negative.
#
```python
min_dist
```
 The minimum distance between embedded points. It must be non-negative and less than or equal to ```self.spread```.
#
```
spread
```
Determines the scale at which embedded points will be spread out. Higher values will lead to more separation between points.
#
```
negative_sample_rate
```
The rate at which to sample negative examples during local optimization. It must be positive.
#
```
global_learning_rate
```
The learning rate for the global optimization phase. It must be positive.
#
```
local_learning_rate
```
The learning rate for the local optimization phase. It must be positive.
#
```
n_neighbors
```
The size of the local neighborhood (defined by the number of nearby sample points) used for manifold approximation. Bigger values lead to a more comprehensive view of the manifold, whereas smaller values retain more local information. Generally, values should fall within the range of 2 to 100. It must be an integer greater than 1.
#
```
hub_num
```
Number of hub points to use for the embedding. It must be a positive integer or -1 (None).
#
```
n_components
```
The dimensionality of the output embedding space. It must be a positive integer.This defaults to 2 to provide easy visualization, but can reasonably be set to any integer value in the range 2 to 100.
#
```
global_n_epochs
```
The number of epochs for the global optimization phase. It must be a positive integer of at least 10.
#
```
local_n_epochs
```
The number of epochs for the local optimization phase. It must be a positive integer of at least 10.
#
```
_sparse_data
```
A boolean flag indicating whether the input data is sparse (True) or dense (False).
#
```
_input_distance_func
```
The input distance function used for the nearest neighbor search. It is determined based on the specified ```self.metric``` and whether the data is sparse or dense.
#
```
angular_rp_forest
```

A boolean flag that indicates whether to use angular random projection forest for approximate nearest neighbor search. It is set to True if the ```self.metric``` is in the set ```{"cosine", "correlation", "dice", "jaccard", "ll_dirichlet", "hellinger"}```.


These parameters, along with their conditions and constraints, control various aspects of the embedding process, including the distance metric, optimization settings, and the structure of the resulting low-dimensional space.

Whether to utilize an angular random projection forest for initializing the approximate nearest neighbor search. This approach can offer improved speed, but it is primarily beneficial for metrics employing an angular-style distance, such as cosine, correlation, and others. For these metrics, angular forests will be automatically selected.

#

# Functions

### fit

```python 
def fit(self, X):
```

This ```fit``` function embeds the input data X into a lower-dimensional space. It handles optional arguments, validates parameters, checks the data sparsity, and builds the nearest neighbor graph structure. It also computes global and local optimization, initializes the embedding using the original hub information, and embeds outliers.

After validating the input data and setting default values for optional arguments, the function checks if the metric is supported by ```PyNNDescent``` and computes the nearest neighbors accordingly. It then builds 

the k-nearest neighbor graph structure, runs global and local optimization, and embeds the remaining data points. Finally, it embeds outliers and returns the fitted model.


+ **Parameters**

``` python
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
```
           
 If the metric is 'precomputed' X must be a square distance matrix. Otherwise it contains a sample per row. If the method is 'exact', X may be a sparse matrix of type 'csr', 'csc' or 'coo'.

#
### fit_transform

```python
def fit_transform(self, X):
```

Fit X into an embedded space and return that transformed output.

+ **Parameters**

```python
Xarray, shape (n_samples, n_features) or (n_samples, n_samples)
```
If the metric is ‘precomputed’ X must be a square distance matrix. Otherwise it contains a sample per row.

+ **Returns**

 ```X_newarray, shape (n_samples, n_components)```
Embedding of the training data in low-dimensional space.


#


# Useful Functions

### build_knn_graph

```python
def build_knn_graph(data, sorted_index, hub_num,):
```
The ```build_knn_graph``` function constructs a k-nearest neighbor (k-NN) graph using a divide-and-conquer approach by selecting hub points from the data. It takes the input data, sorted indices based on the frequency of occurrences in the k-NN graph, and the number of hubs to be used.

+ **Parameters**

```data:``` A 2D array representing the input dataset, with each row being a data point and each column being a feature.

```sorted_index:``` A 1D array containing the indices of data points, sorted by the frequency of their occurrences in the k-NN graph. These indices are sorted in descending order.

```hub_num:``` An integer specifying the number of hubs to be used for constructing the k-NN graph. These hubs serve as the central points for building disjoint sets.

+ **Returns**

The ```build_knn_graph``` function returns a 2D NumPy array called disjoints. Each row of this array represents a disjoint set of data points, and the length of each row is equal to the number of data points divided by the hub_num parameter, rounded up to the nearest integer. The disjoint sets are built by selecting the nearest data points to each hub, based on the Euclidean distance between them.


#

### pick_hubs

```python
def pick_hubs(disjoints, random_state, popular=False,):
```

The ```pick_hubs``` function is responsible for selecting "hub" points from each disjoint set of data points. These hubs serve as representative points for their respective disjoint sets and are used in the subsequent steps of the algorithm.

+ **Parameters**

```disjoints:``` A 2D NumPy array where each row represents a disjoint set of data points.

```random_state:``` A random state object for reproducibility when making random choices.

```popular ```(optional, default is False): A boolean flag indicating whether to pick the most popular (first) element in each disjoint set as the hub or to pick a random element as the hub.

+ **Returns**

The function returns a list called ```hubs```, which contains the indices of the selected hubs. These hub points are used later in the algorithm for various tasks like building the graph, global optimization, and local optimization.
#

### build_global_structure

```python
def build_global_structure( data, hubs, n_components, a, b, random_state, 
alpha=0.0065, n_epochs=30, verbose=False, label=None, init_global="pca", ):
```

The ```build_global_structure``` function is responsible for constructing the global structure of the data using the selected hub points. This global structure serves as an initial embedding for the subsequent steps of the algorithm.

+ **Parameters**

```data:``` A 2D NumPy array containing the input data.

```hubs:``` A list of hub indices, as selected by the ```pick_hubs``` function.

```n_components:``` The number of dimensions for the reduced space (embedding).

```a and b:``` Parameters that control the balance between local and global structure in the embedding.

```random_state:``` A random state object for reproducibility.

```alpha:``` The learning rate for the global optimization step, defaulting to ```0.0065```.

```n_epochs:``` The number of epochs to run the global optimization, defaulting to ```30```.

```verbose:``` A boolean flag that indicates whether to print progress messages.

```label:``` An optional array of labels for the data points, used only if verbose is True.

```init_global:``` The method to initialize the global structure. It can be one of the following strings: 'pca', 'random', or 'spectral'. Alternatively, you can pass a pre-initialized 2D NumPy array.

+ **Returns**

The ```build_global_structure function``` returns the optimized global structure of the input data, which is an embedded representation of the hub points in the lower-dimensional space. This global structure serves as an initial embedding for the subsequent steps of the algorithm.
#

### embed_others_nn

```python
def embed_others_nn(
    data, init_global, hubs, knn_indices, nn_consider, random_state, label, verbose=False,
):
```

The ```embed_others_nn``` function computes the initial embedding for non-hub points using the information from the hub nodes.

+ **Parameters**

```data:``` The input data, a NumPy array of shape (n_samples, n_features).

```init_global:``` The initial global structure, a NumPy array containing the initial embeddings of the hub points.

```hubs:``` A list or array containing the indices of the hub points in the dataset.

```knn_indices:``` A NumPy array containing the k-nearest neighbor indices for each point in the dataset.

```nn_consider:``` An integer specifying the number of nearest neighbors to consider when embedding non-hub points.

```random_state:``` A random state instance for reproducibility.

```label:``` An optional array of labels for each point in the dataset, used for generating plots when verbose is set to True.

```verbose:``` A boolean flag, set to True to display additional information and generate plots during the execution of the function, and False otherwise. The default value is False.


+ **Returns**

```init:``` A NumPy array containing the initial embeddings for all the points, including both the hubs and non-hub points.

```hub_info:``` A NumPy array containing information about the type of each point: hubs are assigned a value of 2, hub nearest neighbors are assigned a value of 1, and outliers are assigned a value of 0.

```hubs:``` The updated list of hub indices after incorporating their nearest neighbors.

#

### embed_outliers

```python
def embed_outliers(
    data, init, hubs, disjoints, random_state, label, verbose=False,
):
```

The ```embed_outliers``` function is responsible for embedding the outlier points in the dataset. 

+ **Parameters**

```data:``` The input data, a NumPy array of shape ```(n_samples, n_features)```.

```init:``` The initial embedding, a NumPy array containing the current embeddings of all points in the dataset except the outliers.

```hubs:``` A list or array containing the indices of the hub points in the dataset.

```disjoints:``` A NumPy array representing the disjoint sets of nearest neighbors for each hub.

```random_state:``` A random state instance for reproducibility.

```label:``` An optional array of labels for each point in the dataset, used for generating plots when verbose is set to True.

```verbose:``` A boolean flag, set to True to display additional information and generate plots during the execution of the function, and False otherwise. The default value is False.

+ **Returns**

The function returns the updated embedding (a NumPy array) that includes the outlier points. This updated embedding represents the positions of all data points in the low-dimensional space, including the hub points, other points, and outliers.
#

### hub_nn_num
 
```python
def hub_nn_num(
    data, hubs, knn_indices, nn_consider=10,
):
```

The ```hub_nn_num``` function is responsible for expanding the set of hub nodes by including their nearest neighbors. It takes a high-dimensional dataset, the current set of hub nodes, and the k-nearest neighbor indices for each data point as input parameters. 

+ **Parameters**

```data:``` A NumPy array representing the high-dimensional dataset.

```hubs:``` A list or array of hub indices.

```knn_indices:``` A NumPy array containing the indices of the k-nearest neighbors for each data point.

```nn_consider``` (optional, default=10): The number of nearest neighbors to consider for each hub node.

+ **Returns**

The ```hub_nn_num``` function returns an updated NumPy array containing the expanded set of hub nodes, which includes the original hub nodes as well as their nearest neighbors (up to the specified number of neighbors to consider, nn_consider). This new set of hub nodes will be used for further processing in the embedding or visualization steps.
#

### nn_initialize

```python
def nn_initialize(
    data, init, original_hubs, hub_nn, random, nn_consider=10,
):
```

The ```nn_initialize``` function initializes the init array based on the nearest neighbors of hub nodes in a dataset and random values. It can be used in various applications, such as clustering or classification tasks, where the initialization of data points is crucial for the algorithm's performance.

+ **Parameters**

```data```: A 2D NumPy array representing the dataset, where rows are data points and columns are features.

```init```: A 1D NumPy array that will be updated during the function execution.
original_hubs: A 1D NumPy array containing the indices of the hub nodes in the data array.

```hub_nn```: A 1D NumPy array containing the indices of the nearest neighbors of the hub nodes in the data array.

```random```: A 1D NumPy array containing random values corresponding to the indices of hub_nn.

```nn_consider```: An optional integer parameter specifying the number of nearest neighbors to consider while updating the values of init. The default value is 10.

+ **Returns**

The code returns the updated ```init``` array after performing the nearest neighbor-based initialization.

The ```init``` array is a 1D NumPy array with the same length as the number of data points in the input dataset ```data```. The values in the ```init``` array are updated based on the nearest neighbors of the hub nodes and random values

#

### select_from_knn

```python
def select_from_knn(
    knn_indices, knn_dists, hub_info, n_neighbors, n,
):
```

```select_from_knn``` is a function that filters the nearest neighbors based on the hub_info array.

+ **Parameters**

```knn_indices```: A 2D NumPy array of shape (n, k) containing the indices of the k-nearest neighbors for each of the n data points.

```knn_dists```: A 2D NumPy array of shape (n, k) containing the distances of the k-nearest neighbors for each of the n data points.

```hub_info```: A 1D NumPy array of length n, where each element represents a data point's "hub" status. A positive value indicates that the point is not an outlier, while a value less than or equal to 0 indicates an outlier.

```n_neighbors```: An integer specifying the number of nearest neighbors to be selected for each data point.

```n```: The number of data points in the dataset.

+ **Returns**

The ```select_from_knn``` function returns three NumPy arrays: ```out_indices```, ```out_dists```, and ```counts```.

```out_indices```: A 2D NumPy array of shape (n, n_neighbors) that contains the filtered indices of the nearest neighbors for each of the n data points. The neighbors are filtered based on the hub_info array to exclude outliers.

```out_dists```: A 2D NumPy array of shape (n, n_neighbors) that contains the corresponding distances of the filtered nearest neighbors for each of the n data points.

```counts```: A 1D NumPy array of length n that stores the number of neighbors selected for each data point. This array can be useful to know how many valid nearest neighbors each data point has after filtering.
#

### apppend_knn

```python
def apppend_knn(
    data, knn_indices, knn_dists, hub_info, n_neighbors, counts, counts_sum,
):
```

The ```apppend_knn``` function is designed to append additional nearest neighbors to the existing nearest neighbors information for each data point if the number of neighbors is less than the specified ```n_neighbors```.

+ **Parameters**

```data```: A 2D NumPy array representing the dataset, where rows are data points and columns are features.

```knn_indices```: A 2D NumPy array containing the indices of the k-nearest neighbors for each data point.

```knn_dists```: A 2D NumPy array containing the distances of the k-nearest neighbors for each data point.

```hub_info```: A 1D NumPy array containing information about the "hub" status of each data point. A positive value indicates that the point is not an outlier, while a value less than or equal to 0 indicates an outlier.

```n_neighbors```: An integer specifying the desired number of nearest neighbors for each data point.

```counts```: A 1D NumPy array containing the number of nearest neighbors currently available for each data point.

```counts_sum```: An integer representing the sum of counts array.

+ **Returns**

The function returns the updated ```knn_indices```, ```knn_dists```, and ```counts_sum```.
#

### local_optimize_nn
```python
def local_optimize_nn( data, graph, hub_info, n_components, learning_rate, a, b, gamma, negative_sample_rate, n_epochs, init, random_state, parallel=False, verbose=False, label=None, ):
```

The local_optimize_nn function is designed to optimize the initial embedding of the data points using a graph-based approach, taking into account the hub information. 

+ **Parameters**

```data```: A 2D NumPy array representing the dataset, where rows are data points and columns are features.

```graph```: A sparse matrix representing the graph, with weights indicating the strength of connections between data points.

```hub_info```: A 1D NumPy array containing information about the "hub" status of each data point. A positive value indicates that the point is not an outlier, while a value less than or equal to 0 indicates an outlier.

```n_components```: The dimension of the target embedding space.

```learning_rate```: The learning rate for the optimization process.

```a, b, gamma```: Hyperparameters controlling the balance between attractive and repulsive forces during optimization.

```negative_sample_rate```: The rate at which negative samples are drawn during optimization.

```n_epochs```: The number of epochs to run the optimization process.

```init```: The initial state of the embedding (either a 1D or 2D NumPy array).

```random_state```: A NumPy random state instance for reproducibility.

```parallel```: A boolean flag indicating whether to run the optimization process in parallel.

```verbose```: A boolean flag indicating whether to print optimization progress.

```label```: An optional label for the optimization process.

+ **Returns**

The ```local_optimize_nn``` function returns a 2D NumPy array called ```embedding```. This array represents the optimized embedding of the data points in the target space. The optimized embedding is calculated using a graph-based approach that takes into account the hub information and adjusts the positions of the data points to better reflect the underlying structure of the data. 

---
[UMATO Github](https://github.com/hyungkwonko/umato)
