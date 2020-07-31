# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import locale
from warnings import warn
import time

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba

import umato.distances as dist

import umato.sparse as sparse
import umato.sparse_nndescent as sparse_nn

from umato.utils import (
    tau_rand_int,
    deheap_sort,
    submatrix,
    ts,
    csr_unique,
    fast_knn_indices,
)
from umato.nndescent import (
    # make_nn_descent,
    # make_initialisations,
    # make_initialized_nnd_search,
    nn_descent,
    initialized_nnd_search,
    initialise_search,
)
from umato.rp_tree import rptree_leaf_array, make_forest
from umato.spectral import spectral_layout
from umato.utils import deheap_sort, submatrix
from umato.layouts import (
    optimize_layout_euclidean,
    optimize_global_layout,
    nn_layout_optimize,
)

try:
    # Use pynndescent, if installed (python 3 only)
    from pynndescent import NNDescent
    from pynndescent.distances import named_distances as pynn_named_distances
    from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances

    _HAVE_PYNNDESCENT = True
except ImportError:
    _HAVE_PYNNDESCENT = False

import gudhi as gd

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def breadth_first_search(adjmat, start, min_vertices):
    explored = []
    queue = [start]
    levels = {}
    levels[start] = 0
    max_level = np.inf
    visited = [start]

    while queue:
        node = queue.pop(0)
        explored.append(node)
        if max_level == np.inf and len(explored) > min_vertices:
            max_level = max(levels.values())

        if levels[node] + 1 < max_level:
            neighbors = adjmat[node].indices
            for neighbour in neighbors:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)

                    levels[neighbour] = levels[node] + 1

    return np.array(explored)


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def nearest_neighbors(
    X,
    n_neighbors,
    metric,
    metric_kwds,
    angular,
    random_state,
    low_memory=False,
    use_pynndescent=True,
    verbose=False,
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metric: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    low_memory: bool (optional, default False)
        Whether to pursue lower memory NNdescent.

    verbose: bool (optional, default False)
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    rp_forest: list of trees
        The random projection forest used for searching (if used, None otherwise)
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = fast_knn_indices(X, n_neighbors)
        # knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

        rp_forest = []
    else:
        # TODO: Hacked values for now
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        if _HAVE_PYNNDESCENT and use_pynndescent:
            nnd = NNDescent(
                X,
                n_neighbors=n_neighbors,
                metric=metric,
                metric_kwds=metric_kwds,
                random_state=random_state,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                low_memory=low_memory,
                verbose=verbose,
            )
            knn_indices, knn_dists = nnd.neighbor_graph
            rp_forest = nnd
        else:
            # Otherwise fall back to nn descent in umap
            if callable(metric):
                _distance_func = metric
            elif metric in dist.named_distances:
                _distance_func = dist.named_distances[metric]
            else:
                raise ValueError("Metric is neither callable, nor a recognised string")

            rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

            if scipy.sparse.isspmatrix_csr(X):
                if callable(metric):
                    _distance_func = metric
                else:
                    try:
                        _distance_func = sparse.sparse_named_distances[metric]
                        if metric in sparse.sparse_need_n_features:
                            metric_kwds["n_features"] = X.shape[1]
                    except KeyError as e:
                        raise ValueError(
                            "Metric {} not supported for sparse data".format(metric)
                        ) from e

                # Create a partial function for distances with arguments
                if len(metric_kwds) > 0:
                    dist_args = tuple(metric_kwds.values())

                    @numba.njit()
                    def _partial_dist_func(ind1, data1, ind2, data2):
                        return _distance_func(ind1, data1, ind2, data2, *dist_args)

                    distance_func = _partial_dist_func
                else:
                    distance_func = _distance_func
                # metric_nn_descent = sparse.make_sparse_nn_descent(
                #     distance_func, tuple(metric_kwds.values())
                # )

                if verbose:
                    print(ts(), "Building RP forest with", str(n_trees), "trees")

                rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
                leaf_array = rptree_leaf_array(rp_forest)

                if verbose:
                    print(ts(), "NN descent for", str(n_iters), "iterations")
                knn_indices, knn_dists = sparse_nn.sparse_nn_descent(
                    X.indices,
                    X.indptr,
                    X.data,
                    X.shape[0],
                    n_neighbors,
                    rng_state,
                    max_candidates=60,
                    sparse_dist=distance_func,
                    low_memory=low_memory,
                    rp_tree_init=True,
                    leaf_array=leaf_array,
                    n_iters=n_iters,
                    verbose=verbose,
                )
            else:
                # metric_nn_descent = make_nn_descent(
                #     distance_func, tuple(metric_kwds.values())
                # )
                if len(metric_kwds) > 0:
                    dist_args = tuple(metric_kwds.values())

                    @numba.njit()
                    def _partial_dist_func(x, y):
                        return _distance_func(x, y, *dist_args)

                    distance_func = _partial_dist_func
                else:
                    distance_func = _distance_func

                if verbose:
                    print(ts(), "Building RP forest with", str(n_trees), "trees")

                rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
                leaf_array = rptree_leaf_array(rp_forest)  # stacked rp_tree indices

                if verbose:
                    print(ts(), "NN descent for", str(n_iters), "iterations")
                knn_indices, knn_dists = nn_descent(
                    X,
                    n_neighbors,
                    rng_state,
                    max_candidates=60,
                    dist=distance_func,
                    low_memory=low_memory,
                    rp_tree_init=True,
                    leaf_array=leaf_array,
                    n_iters=n_iters,
                    verbose=verbose,
                )

            if np.any(knn_indices < 0):
                warn(
                    "Failed to correctly find n_neighbors for some samples."
                    "Results may be less than ideal. Try re-running with"
                    "different parameters."
                )
    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, rp_forest


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths2(knn_indices, knn_dists, sigmas, rhos, hubs):

    n_samples = len(hubs)
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(n_samples * n_neighbors, dtype=np.int32)
    cols = np.zeros(n_samples * n_neighbors, dtype=np.int32)
    vals = np.zeros(n_samples * n_neighbors, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = hubs[i]
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    hubs=None,
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean (or l2)
            * manhattan (or l1)
            * cityblock
            * braycurtis
            * canberra
            * chebyshev
            * correlation
            * cosine
            * dice
            * hamming
            * jaccard
            * kulsinski
            * ll_dirichlet
            * mahalanobis
            * matching
            * minkowski
            * rogerstanimoto
            * russellrao
            * seuclidean
            * sokalmichener
            * sokalsneath
            * sqeuclidean
            * yule
            * wminkowski

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=verbose
        )

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    if hubs is not None:  # build graph using only (hub + nn) nodes
        rows, cols, vals = compute_membership_strengths2(
            knn_indices, knn_dists, sigmas, rhos, hubs,
        )
    else:
        rows, cols, vals = compute_membership_strengths(
            knn_indices, knn_dists, sigmas, rhos
        )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols))
    )  # (TODO) do I need to set the shape ?
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    return result, sigmas, rhos


@numba.njit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.

    target: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist float (optional, default 5.0)
        The distance between unmatched labels.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if (target[i] == -1) or (target[j] == -1):
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)

    return


@numba.jit()
def fast_metric_intersection(
    rows, cols, values, discrete_space, metric, metric_args, scale
):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array of shape
        An array of the values of each non-zero in the sparse matrix
        representation.

    discrete_space: array of shape (n_samples, n_features)
        The vectors of categorical labels to use in the intersection.

    metric: numba function
        The function used to calculate distance over the target array.

    scale: float
        A scaling to apply to the metric.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        dist = metric(discrete_space[i], discrete_space[j], *metric_args)
        values[nz] *= np.exp(-(scale * dist))

    return


@numba.njit()
def reprocess_row(probabilities, k=15, n_iters=32):
    target = np.log2(k)

    lo = 0.0
    hi = NPY_INFINITY
    mid = 1.0

    for n in range(n_iters):

        psum = 0.0
        for j in range(probabilities.shape[0]):
            psum += pow(probabilities[j], mid)

        if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
            break

        if psum < target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0

    return np.power(probabilities, mid)


@numba.njit()
def reset_local_metrics(simplicial_set_indptr, simplicial_set_data):
    for i in range(simplicial_set_indptr.shape[0] - 1):
        simplicial_set_data[
            simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]
        ] = reprocess_row(
            simplicial_set_data[simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]]
        )
    return


def reset_local_connectivity(simplicial_set, reset_local_metric=False):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.

    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.

    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    if reset_local_metric:
        simplicial_set = simplicial_set.tocsr()
        reset_local_metrics(simplicial_set.indptr, simplicial_set.data)
        simplicial_set = simplicial_set.tocoo()
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()

    return simplicial_set


def discrete_metric_simplicial_set_intersection(
    simplicial_set,
    discrete_space,
    unknown_dist=1.0,
    far_dist=5.0,
    metric=None,
    metric_kws={},
    metric_scale=1.0,
):
    """Combine a fuzzy simplicial set with another fuzzy simplicial set
    generated from discrete metric data using discrete distances. The target
    data is assumed to be categorical label data (a vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.

    TODO: optional category cardinality based weighting of distance

    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.

    discrete_space: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist: float (optional, default 5.0)
        The distance between unmatched labels.

    metric: str (optional, default None)
        If not None, then use this metric to determine the
        distance between values.

    metric_scale: float (optional, default 1.0)
        If using a custom metric scale the distance values by
        this value -- this controls the weighting of the
        intersection. Larger values weight more toward target.

    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    if metric is not None:
        # We presume target is now a 2d array, with each row being a
        # vector of target info
        if metric in dist.named_distances:
            metric_func = dist.named_distances[metric]
        else:
            raise ValueError("Discrete intersection metric is not recognized")

        fast_metric_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            metric_func,
            tuple(metric_kws.values()),
            metric_scale,
        )
    else:
        fast_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            unknown_dist,
            far_dist,
        )

    simplicial_set.eliminate_zeros()

    return reset_local_connectivity(simplicial_set)


def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight):

    result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_intersection(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
        weight,
    )

    return result


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


############### Hyung-Kwon Ko
############### Hyung-Kwon Ko
############### Hyung-Kwon Ko


def plot_tmptmp(data, label, name):
    import matplotlib.pyplot as plt

    plt.scatter(data[:, 0], data[:, 1], s=2.0, c=label, cmap="Spectral", alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(11))
    plt.title("Embedded")
    plt.savefig(f"./tmp/{name}.png")
    plt.close()


def check_nn_accuracy(
    indices_info, label,
):
    # ix = np.arange(indices_info.shape[0])
    # ix2 = ix[self.ll < 10]
    scores = np.array([])
    for i in np.arange(indices_info.shape[0]):
        score = 0
        for j in range(1, indices_info.shape[1]):
            if label[indices_info[i][j]] == label[indices_info[i][0]]:
                score += 1.0 / (indices_info.shape[1] - 1)
        scores = np.append(scores, score)
    print(len(scores))
    print(np.mean(scores))
    return 0


# @numba.njit(
#     parallel=True,
#     fastmath=True,
# )
# def distance_calculation(data, sorted_index, source):
#     distances = np.ones(len(sorted_index)) * np.inf
#     for k in numba.prange(len(sorted_index)):
#         distance = 0.0
#         if sorted_index[k] > -1:
#             target = sorted_index[k]
#             for d in numba.prange(data.shape[1]):
#                 distance += (data[source][d] - data[target][d]) ** 2
#             distances[target] = np.sqrt(distance)
#     return distances

# @numba.njit(
#     parallel=True,
#     fastmath=True,
# )
# def dim_calculation(data, source, target):
#     distance = 0.0
#     for d in numba.prange(data.shape[1]):
#         distance += (data[source][d] - data[target][d]) ** 2
#     return np.sqrt(distance)


@numba.njit(
    # parallel=True,  # can SABOTAGE the array order (should be used with care)
    fastmath=True,
)
def disjoint_nn(
    data, sorted_index, hub_num,
):
    sorted_index_c = sorted_index.copy()

    leaf_num = int(np.ceil(data.shape[0] / hub_num))
    disjoints = []

    for i in range(hub_num):
        tmp = 0
        source = -1
        disjoint = []

        # append the first element
        for j in range(len(sorted_index_c)):
            if sorted_index_c[j] > -1:
                source = sorted_index_c[j]
                disjoint.append(source)
                sorted_index_c[j] = -1
                tmp += 1
                break
        if source == -1:
            break  # break if all indices == -1

        # get distance for each element
        distances = np.ones(len(sorted_index_c)) * np.inf
        for k in range(len(sorted_index_c)):
            distance = 0.0
            if sorted_index_c[k] > -1:
                target = sorted_index_c[k]
                for d in range(data.shape[1]):
                    distance += (data[source][d] - data[target][d]) ** 2
                distances[target] = np.sqrt(distance)

        # append other elements
        for _ in range(leaf_num - 1):
            val = min(distances)
            if np.isinf(val):
                disjoint = disjoint + [-1] * (leaf_num - tmp)
                break
            else:
                min_index = np.argmin(distances)
                disjoint.append(min_index)
                distances[min_index] = np.inf
                sorted_index_c[sorted_index_c == min_index] = -1
                tmp += 1

        disjoints.append(disjoint)

    return np.array(disjoints)


# @numba.njit()
def pick_hubs(
    disjoints, random_state, popular=False,
):
    if popular:
        return disjoints[:, 0]
    else:
        hubs = []
        (hub_num, _) = disjoints.shape

        # append until second to last element
        for i in range(hub_num - 1):
            choice = random_state.choice(disjoints[i])
            hubs.append(choice)

        # append last element
        last = disjoints[hub_num - 1]
        last = last[last != -1]
        choice = random_state.choice(last)
        hubs.append(choice)

        if hub_num != len(hubs):
            ValueError(f"hub_num({hub_num}) is not the same as hubs({hubs})!")

        return hubs


def hub_candidates(
    data, sorted_index, random_state, hub_num, iter_num=5,
):
    hubs_list = []
    disjoints = disjoint_nn(data=data, sorted_index=sorted_index, hub_num=hub_num,)

    for i in range(iter_num):
        hubs = pick_hubs(disjoints=disjoints, random_state=random_state, popular=False,)
        hubs_list.append(hubs)

    return hubs_list, disjoints


def get_homology(data, local_knum, top_num, random_state):
    dist = euclidean_distances(data, data)
    dist /= dist.max()

    nn_index = np.argpartition(dist, kth=local_knum - 1, axis=-1)[
        :, :local_knum
    ]  # kill using local connectivity

    for i in range(len(dist)):
        dist[i][nn_index[i]] = random_state.random(local_knum) * 0.1

    rc = gd.RipsComplex(distance_matrix=dist, max_edge_length=1.0)
    rc_tree = rc.create_simplex_tree(max_dimension=2)
    barcodes = rc_tree.persistence()

    hom1 = rc_tree.persistence_intervals_in_dimension(1)
    # cutoff = 0.3
    # hom1 = hom1[np.where(abs(hom1[:,0] - hom1[:,1]) > cutoff)]
    hom1_max = abs(hom1[:, 1] - hom1[:, 0])
    hom1_max_ix = hom1_max.argsort()[-top_num:][::-1]

    return hom1[hom1_max_ix]


def select_hubs_homology(
    data,
    random_state,
    sorted_index,
    hub_num=50,
    iter_num=5,
    interval=50,
    top_num=30,
    cutoff=0.05,
    local_knum=7,
):
    print("[INFO]: Select hub nodes using homology")

    while True:
        results = []
        k1_list = []
        k2_list = []

        hubs_list, disjoints = hub_candidates(
            data, sorted_index, random_state, hub_num, iter_num
        )

        hubs_list2, _ = hub_candidates(
            data, sorted_index, random_state, hub_num + interval, iter_num
        )

        for i in range(iter_num):
            d1 = data[hubs_list[i]]
            k1 = get_homology(d1, local_knum, top_num, random_state)
            k1_list.append(k1)

            d2 = data[hubs_list2[i]]
            k2 = get_homology(d2, local_knum, top_num, random_state)
            k2_list.append(k2)

        for _k1 in k1_list:
            for _k2 in k2_list:
                # result = gd.bottleneck_distance(_k1, _k2, 0.01)  # approximation
                result = gd.bottleneck_distance(_k1, _k2)
                results.append(result)

        val = np.mean(results)
        print(f"val: {val}")

        if val < cutoff:
            break
        elif hub_num > 300:  # break if > 300
            warn(f"Hub node number set to {hub_num}!")
            break
        else:
            hub_num += interval

    hubs = pick_hubs(disjoints=disjoints, random_state=random_state, popular=True,)

    return hubs, disjoints


def remove_local_connect(array, random_state, loc=0.05, num=-1):
    if num < 0:
        num = array.shape[0] // 10  # use 10 % of the hub nodes

    normal = random_state.normal(loc=loc, scale=loc, size=num).astype(np.float32)
    normal = np.clip(normal, a_min=0.0, a_max=loc * 2)

    for _, e in enumerate(array):
        indices = np.argsort(e)[:num]
        e[indices] = np.sort(normal)

    return array


def build_global_structure(
    data,
    hubs,
    n_components,
    a,
    b,
    random_state,
    alpha=0.0055,
    max_iter=10,
    verbose=False,
    label=None,
    init_global="pca",
):
    print("[INFO] Building global structure")

    if init_global == "pca":
        Z = PCA(n_components=n_components).fit_transform(data[hubs])
        Z /= Z.max()
    elif init_global == "random":
        Z = np.random.random((len(hubs), n_components))
    else:
        raise ValueError("Check hub node initializing method!")

    P = euclidean_distances(data[hubs])
    # P /= np.sum(P, axis=1, keepdims=True)
    P /= P.max()

    # local connectivity for global optimization
    # P = remove_local_connect(P, random_state)

    # Z = (1.0 * (Z - np.min(Z, 0)) / (np.max(Z, 0) - np.min(Z, 0))).astype(
    #     np.float32, order="C"
    # )

    if verbose:
        result = optimize_global_layout(
            P=P,
            Z=Z,
            a=a,
            b=b,
            alpha=alpha,
            max_iter=max_iter,
            verbose=True,
            savefig=True,
            label=label[hubs],
        )
    else:
        result = optimize_global_layout(
            P, Z, a, b, alpha=alpha, max_iter=max_iter
        )  # (TODO) how to optimize max_iter & alpha?

    return result


def embed_others_nn(
    data, init_global, hubs, knn_indices, random_state, label,
):
    init = np.zeros((data.shape[0], init_global.shape[1]))
    original_hubs = hubs.copy()
    init[original_hubs] = init_global

    while True:
        val = len(hubs)
        hubs = hub_nn_num(
            data=data, hubs=hubs, knn_indices=knn_indices, nn_consider=10,
        )

        if val == len(hubs):
            if len(init) > len(hubs):
                print(f"len(hubs) {len(hubs)} is smaller than len(init) {len(init)}")
            break

    # generate random normal distribution
    random_normal = random_state.normal(
        loc=0.0, scale=0.05, size=list(init.shape)
    ).astype(np.float32)

    hub_nn = set(hubs) - set(original_hubs)
    hub_nn = np.array(list(hub_nn))

    # initialize other nodes' position using only hub information
    init = nn_initialize(
        data=data,
        init=init,
        original_hubs=original_hubs,
        hub_nn=hub_nn,
        random=random_normal,
        nn_consider=10,
    )

    # np.array of hub information (hubs = 2, hub_nn = 1, outliers = 0)
    hub_info = np.zeros(data.shape[0])
    hub_info[hub_nn] = 1
    hub_info[original_hubs] = 2

    # save figure2
    plot_tmptmp(data=init[hubs], label=label[hubs], name=f"pic2")

    return init, hub_info, hubs


def embed_others_disjoint(
    data, init, hubs, disjoints, random_state, label,
):
    # generate random normal distribution
    random_normal = random_state.normal(scale=0.02, size=list(init.shape)).astype(
        np.float32
    )

    # append other nodes using NN disjoint information
    init, nodes_number = disjoint_initialize(
        data=data, init=init, hubs=hubs, disjoints=disjoints, random=random_normal,
    )

    if len(init) != len(nodes_number):
        raise ValueError(
            f"total data # ({len(init)}) != total embedded # ({len(nodes_number)})!"
        )

    # save figure3
    plot_tmptmp(data=init, label=label, name="pic4_disjoint")

    return init


@numba.njit()
def disjoint_initialize(
    data, init, hubs, disjoints, random,
):

    hubs_true = np.zeros(data.shape[0])
    hubs_true[hubs] = True
    hubs = set(hubs)

    nndist = np.sum(init[:, 1]) / len(hubs)

    for disjoint in disjoints:
        for j in disjoint:
            # j == -1 means we've run all the iteration
            if j == -1:
                break
            # if it is not a hub node, we should embed this using NN in disjoint set
            if not hubs_true[j]:
                distances = []
                indices = []
                # we use its neighbors
                for k in disjoint:
                    if hubs_true[k]:
                        distance = 0.0
                        for l in range(data.shape[1]):
                            distance += (data[j][l] - data[k][l]) ** 2
                        distance = np.sqrt(distance)
                        distances.append(distance)
                        indices.append(k)
                ix = np.array(distances).argsort()[0]
                target_ix = indices[ix]
                init[j] = init[target_ix] + random[j]  # add random value

                hubs.add(j)

    return init, hubs


@numba.njit()
def hub_nn_num(
    data, hubs, knn_indices, nn_consider=10,
):
    print("[INFO] get hub_nn indices")

    num_log = np.zeros(data.shape[0])
    num_log[hubs] = -1

    hubs = set(hubs)
    hubs_fin = hubs.copy()

    for i in hubs:
        for j, e in enumerate(knn_indices[i]):
            if j > nn_consider:
                break
            if num_log[e] > -1:
                hubs_fin.add(e)

    return np.array(list(hubs_fin))


@numba.njit(
    locals={
        "num_log": numba.types.float32[::1],
        "index": numba.types.int32,
        "dists": numba.types.float32[::1],
        "dist": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def nn_initialize(
    data, init, original_hubs, hub_nn, random, nn_consider=10,
):
    print(
        "[INFO] Embedding other nodes using NN information using only original hub information"
    )

    num_log = np.zeros(data.shape[0], dtype=np.float32)
    num_log[original_hubs] = -1
    num_log[hub_nn] = -1

    for i in numba.prange(len(hub_nn)):
        # find nearest hub nodes
        dists = np.zeros(len(original_hubs), dtype=np.float32)
        for j in numba.prange(len(original_hubs)):
            dist = 0.0
            for d in numba.prange(data.shape[1]):
                e = original_hubs[j]
                dist += (data[e][d] - data[hub_nn[i]][d]) ** 2
            dists[j] = dist

        # sorted hub indices
        dists_arg = dists.argsort(kind="quicksort")

        for k in numba.prange(nn_consider):
            index = original_hubs[dists_arg[k]]
            init[hub_nn[i]] += init[index]
            num_log[hub_nn[i]] += 1

        # add random value before break
        init[hub_nn[i]] += random[hub_nn[i]]

    for l in numba.prange(data.shape[0]):
        if num_log[l] > 0:
            init[l] /= num_log[l]

    return init


# @numba.njit()
# def remove_from_graph(data, array, hub_info, remove_target):
#     """
#     remove_target == 0: outliers
#     remove_target == 1: NNs
#     remove_target == 2: hubs
#     """
#     for target in remove_target:
#         if target not in [0, 1, 2]:
#             raise ValueError(
#                 "remove_target should be 0 (outliers) or 1 (NNs) or 2 (hubs)"
#             )

#         for i, e in enumerate(array):
#             if hub_info[e] == target:
#                 data[i] = 0

#     return data

# @numba.njit()
# def change_graph_ix(array, hubs):
#     result = array.copy()
#     for i, hub in enumerate(hubs):
#         result[array == i] = hub
#     return result

# @numba.njit(
#     locals={
#         "sigmas": numba.types.float32[::1],
#         "rhos": numba.types.float32[::1],
#         "vals": numba.types.float32[::1],
#         "dists": numba.types.float32[::1],
#     },
#     parallel=True,
#     fastmath=True,
# )
# def fast_knn_indices_hub(X, n_neighbors, hubs, nns):

#     nn_num = len(nns)
#     hub_num = len(hubs)

#     rows = np.zeros(n_neighbors * nn_num, dtype=np.int32)
#     cols = np.zeros(n_neighbors * nn_num, dtype=np.int32)
#     vals = np.zeros(n_neighbors * nn_num, dtype=np.float32)

#     for i in numba.prange(nn_num):
#         dists = np.zeros(hub_num, dtype=np.float32)
#         for j in numba.prange(hub_num):
#             dist = 0.0
#             for d in numba.prange(X.shape[1]):
#                 dist += (X[nns[i]][d] - X[hubs[j]][d]) ** 2
#             dists[j] = np.sqrt(dist)

#         sorted_dists = dists.argsort(kind="quicksort")
#         neighbors = sorted_dists[:n_neighbors]

#         rows[i * n_neighbors : (i + 1) * n_neighbors] = nns[i]
#         cols[i * n_neighbors : (i + 1) * n_neighbors] = neighbors
#         vals[i * n_neighbors : (i + 1) * n_neighbors] = 1.0

#     return rows, cols, vals

# def compute_hub_nn_graph(
#     data, n_neighbors, hub_info,
# ):

#     hubs = np.where(hub_info == 2)[0]
#     nns = np.where(hub_info == 1)[0]
#     knn_indices = fast_knn_indices_hub(data, n_neighbors, hubs, nns)

#     return knn_indices


@numba.njit(
    locals={
        "out_indices": numba.types.int32[:, ::1],
        "out_dists": numba.types.float32[:, ::1],
        "counts": numba.types.int32[::1],
    },
    parallel=True,
    fastmath=True,
)
def select_from_knn(
    knn_indices, knn_dists, hub_info, n_neighbors, n,
):
    out_indices = np.zeros((n, n_neighbors), dtype=np.int32)
    out_dists = np.zeros((n, n_neighbors), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)

    for i in numba.prange(knn_indices.shape[0]):
        if hub_info[i] > 0:
            for j in numba.prange(knn_indices.shape[1]):
                # append directly if it is not an outlier
                if hub_info[knn_indices[i, j]] > 0:
                    out_indices[i, counts[i]] = knn_indices[i, j]
                    out_dists[i, counts[i]] = knn_dists[i, j]
                    counts[i] += 1
                if counts[i] == n_neighbors:
                    break

    return out_indices, out_dists, counts


@numba.njit(
    locals={"dists": numba.types.float32[::1],}, parallel=True, fastmath=True,
)
def apppend_knn(
    data, knn_indices, knn_dists, hub_info, n_neighbors, counts, counts_sum,
):
    for i in numba.prange(data.shape[0]):
        num = n_neighbors - counts[i]
        if hub_info[i] > 0 and num > 0:
            neighbors = knn_indices[i][
                : counts[i]
            ]  # found neighbors (# of neighbors < n_neighbors)

            # find unique target indices
            indices = set()
            for ci in range(
                counts[i]
            ):  # cannot use numba.prange; malloc error occurs... don't know why
                cx = neighbors[ci]
                for cy in range(counts[cx]):
                    indices.add(knn_indices[cx][cy])

            # get target indices
            targets = indices - set(neighbors)
            targets = np.array(list(targets))

            # if there is not enough target, it is a corner case (raise error)
            if len(targets) < num:
                return knn_indices, knn_dists, -1
            else:
                # calculate distances
                dists = np.zeros(len(targets), dtype=np.float32)
                for k in numba.prange(len(targets)):
                    dist = 0.0
                    for d in numba.prange(data.shape[1]):
                        dist += (data[i][d] - data[targets[k]][d]) ** 2
                    dists[k] = np.sqrt(dist)
                sorted_dists_index = dists.argsort(kind="quicksort")

                # add more knns
                for j in numba.prange(num):
                    knn_indices[i][counts[i] + j] = targets[
                        sorted_dists_index[counts[i] + j]
                    ]
                    knn_dists[i][counts[i] + j] = dists[
                        sorted_dists_index[counts[i] + j]
                    ]

                # re-sort index
                sorted_knn_index = knn_dists[i].argsort(kind="quicksort")
                knn_indices[i] = knn_indices[i][sorted_knn_index]
                knn_dists[i] = knn_dists[i][sorted_knn_index]

                # for double check
                counts_sum -= 1

    return knn_indices, knn_dists, counts_sum


def local_optimize_nn(
    data,
    graph,
    hub_info,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    parallel=False,
    verbose=False,
    label=None,
):

    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    print(len(graph.data))

    graph.data[
        hub_info[graph.col] == 2
    ] = 1.0  # current (NNs) -- other (hubs): 1.0 weight
    graph.data[
        hub_info[graph.row] == 2
    ] = 0.0  # current (hubs) -- other (hubs, nns): 0.0 weight (remove)
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    # graph.data[graph.data < 0.2] = 0.0
    graph.eliminate_zeros()

    print(len(graph.data))

    # check_nn_accuracy(indices_info=hub_knn_indices, label=label)

    init_data = np.array(init)
    if len(init_data.shape) == 2:
        if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
            tree = KDTree(init_data)
            dist, ind = tree.query(init_data, k=2)
            nndist = np.mean(dist[:, 1])
            embedding = init_data + random_state.normal(
                scale=0.001 * nndist, size=init_data.shape
            ).astype(np.float32)
        else:
            embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    embedding = (
        20.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    embedding = nn_layout_optimize(
        embedding,
        embedding,
        head,
        tail,
        hub_info,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        initial_alpha,
        negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
        label=label,
    )

    return embedding


class UMATO(BaseEstimator):
    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        hub_num=-1,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        force_approximation_algorithm=False,
        verbose=False,
        unique=False,
        ll=None,
    ):
        self.n_neighbors = n_neighbors
        self.hub_num = hub_num
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique
        self.ll = ll

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random"):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if not isinstance(self.hub_num, int) or self.hub_num < -1:
            raise ValueError("hub_num must be a positive integer or -1 (None)")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs <= 10 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a positive integer of at least 10")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        if self.target_metric_kwds is None:
            self._target_metric_kwds = {}
        else:
            self._target_metric_kwds = self.target_metric_kwds
        # check sparsity of data upfront to set proper _input_distance_func &
        # save repeated checks later on
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        # set input distance metric & inverse_transform distance metric
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(
                self.metric, self._metric_kwds, self._raw_data
            )
            if in_returns_grad:
                _m = self.metric

                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]

                self._input_distance_func = _dist_only
                self._inverse_distance_func = self.metric
            else:
                self._input_distance_func = self.metric
                self._inverse_distance_func = None
                warn(
                    "custom distance metric does not return gradient; inverse_transform will be unavailable. "
                    "To enable using inverse_transform method method, define a distance function that returns "
                    "a tuple of (distance [float], gradient [np.array])"
                )
        elif self.metric == "precomputed":
            if self.unique:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn(
                "using precomputed metric; transform will be unavailable for new data and inverse_transform "
                "will be unavailable for all data"
            )
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in dist.named_distances:
            if self._sparse_data:
                if self.metric in sparse.sparse_named_distances:
                    self._input_distance_func = sparse.sparse_named_distances[
                        self.metric
                    ]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = dist.named_distances[self.metric]
            try:
                self._inverse_distance_func = dist.named_distances_with_gradients[
                    self.metric
                ]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")
        # set ooutput distance metric
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(
                self.output_metric, self._output_metric_kwds
            )
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError(
                    "custom output_metric must return a tuple of (distance [float], gradient [np.array])"
                )
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in dist.named_distances_with_gradients:
            self._output_distance_func = dist.named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in dist.named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )
        # set angularity for NN search based on metric
        if self.metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
            "ll_dirichlet",
            "hellinger",
        ):
            self.angular_rp_forest = True

    def _check_custom_metric(self, metric, kwds, data=None):
        # quickly check to determine whether user-defined
        # self.metric/self.output_metric returns both distance and gradient
        if data is not None:
            # if checking the high-dimensional distance metric, test directly on
            # input data so we don't risk violating any assumptions potentially
            # hard-coded in the metric (e.g., bounded; non-negative)
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            # if checking the manifold distance metric, simulate some data on a
            # reasonable interval with output dimensionality
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))

        if scipy.sparse.issparse(data):
            metric_out = metric(x.indices, x.data, y.indices, y.data, **kwds)
        else:
            metric_out = metric(x, y, **kwds)
        # True if metric returns iterable of length 2, False otherwise
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        if self.verbose:
            print(str(self))

        # Check if we should unique the data
        # We've already ensured that we aren't in the precomputed case
        if self.unique:
            # check if the matrix is dense
            if self._sparse_data:
                # Call a sparse unique function
                index, inverse, counts = csr_unique(X)
            else:
                index, inverse, counts = np.unique(
                    X,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    axis=0,
                )[1:4]
            if self.verbose:
                print(
                    "Unique=True -> Number of data points reduced from ",
                    X.shape[0],
                    " to ",
                    X[index].shape[0],
                )
                most_common = np.argmax(counts)
                print(
                    "Most common duplicate is",
                    index[most_common],
                    " with a count of ",
                    counts[most_common],
                )
        # If we aren't asking for unique use the full index.
        # This will save special cases later.
        else:
            index = list(range(X.shape[0]))
            inverse = list(range(X.shape[0]))

        # Error check n_neighbors based on data size
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X[index].shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        # Note: unless it causes issues for setting 'index', could move this to
        # initial sparsity check above
        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        # pass string identifier if pynndescent also defines distance metric
        if _HAVE_PYNNDESCENT:
            if self._sparse_data and self.metric in pynn_sparse_named_distances:
                nn_metric = self.metric
            elif not self._sparse_data and self.metric in pynn_named_distances:
                nn_metric = self.metric
            else:
                nn_metric = self._input_distance_func
        else:
            nn_metric = self._input_distance_func

        (self._knn_indices, self._knn_dists, _) = nearest_neighbors(
            X[index],
            self._n_neighbors,
            # int(self._n_neighbors * 1.2),  # we can use more neighbors
            nn_metric,
            self._metric_kwds,
            self.angular_rp_forest,
            random_state,
            self.low_memory,
            use_pynndescent=True,
            verbose=self.verbose,
        )

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print(ts(), "Construct global structure")

        ###### Hyung-Kwon Ko
        ###### Hyung-Kwon Ko
        ###### Hyung-Kwon Ko

        print("1: ", ts())

        flat_indices = self._knn_indices.flatten()  # flattening all knn indices
        index, freq = np.unique(flat_indices, return_counts=True)
        # sorted_index = index[freq.argsort(kind="stable")]  # sorted index in increasing order
        sorted_index = index[
            freq.argsort(kind="stable")[::-1]
        ]  # sorted index in decreasing order

        print("2: ", ts())

        t1 = time.time()

        if self.hub_num < 0:
            hubs, disjoints = select_hubs_homology(
                data=X,
                random_state=random_state,
                sorted_index=sorted_index,
                hub_num=50,
            )
        else:
            # get disjoint NN matrix
            disjoints = disjoint_nn(
                data=X, sorted_index=sorted_index, hub_num=self.hub_num,
            )
            # get hub indices from disjoint set
            hubs = pick_hubs(
                disjoints=disjoints, random_state=random_state, popular=True,
            )

        print("3: ", ts())

        # # check NN accuracy
        # check_nn_accuracy(
        #     indices_info=disjoints, label=self.ll,
        # )

        print("4: ", ts())

        init_global = build_global_structure(
            data=X,
            hubs=hubs,
            n_components=self.n_components,
            a=self._a,
            b=self._b,
            random_state=random_state,
            alpha=0.006,
            max_iter=10,
            # verbose=False,
            verbose=True,
            label=self.ll,
        )

        print("5: ", ts())

        init, hub_info, hubs = embed_others_nn(
            data=X,
            init_global=init_global,
            hubs=hubs,
            knn_indices=self._knn_indices,
            random_state=random_state,
            label=self.ll,
        )

        print("6: ", ts())

        self._knn_indices, self._knn_dists, counts = select_from_knn(
            knn_indices=self._knn_indices,
            knn_dists=self._knn_dists,
            hub_info=hub_info,
            n_neighbors=self.n_neighbors,
            n=X.shape[0],
        )

        print("7: ", ts())

        counts_hub = counts[hubs]
        counts_sum = len(counts_hub[counts_hub < self.n_neighbors])
        if counts_sum > 0:
            if self.verbose:
                print(ts(), "Adding more KNNs to build the graph")

            self._knn_indices, self._knn_dists, counts_sum = apppend_knn(
                data=X,
                knn_indices=self._knn_indices,
                knn_dists=self._knn_dists,
                hub_info=hub_info,
                n_neighbors=self.n_neighbors,
                counts=counts,
                counts_sum=counts_sum,
            )

            if counts_sum != 0:
                raise ValueError(
                    f"KNN indices not fully determined! counts_sum: {counts_sum} != 0"
                )

        # check_nn_accuracy(
        #     indices_info=self._knn_indices[hubs], label=self.ll,
        # )

        print("8: ", ts())

        self.graph_, _, _ = fuzzy_simplicial_set(
            X[hubs],
            self.n_neighbors,
            random_state,
            nn_metric,
            self._metric_kwds,
            hubs,
            self._knn_indices[hubs],
            self._knn_dists[hubs],
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
            True,
        )

        print("9: ", ts())

        if self.verbose:
            print(ts(), "Construct local structure")

        with open("./hubs.npy", "wb") as f:
            np.save(f, hubs)

        init = local_optimize_nn(
            data=X,
            graph=self.graph_,
            hub_info=hub_info,
            n_components=self.n_components,
            initial_alpha=self._initial_alpha,
            a=self._a,
            b=self._b,
            gamma=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=n_epochs,
            init=init,
            random_state=random_state,
            metric=self._input_distance_func,
            metric_kwds=self.metric_kwds,
            parallel=False,
            verbose=True,
            label=self.ll,
        )

        print("10: ", ts())

        self.embedding_ = embed_others_disjoint(
            data=X,
            init=init,
            hubs=hubs,
            disjoints=disjoints,
            random_state=random_state,
            label=self.ll,
        )

        print("11: ", ts())

        exit()

        if self.verbose:
            print(ts() + " Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)

        return self

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X, y)
        return self.embedding_
