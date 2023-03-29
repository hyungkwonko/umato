"""
Auhtor: Hyung-Kwon Ko (hkko@hcil.snu.ac.kr)
"""

from __future__ import print_function

import locale
from warnings import warn
import time

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import numba

import umato.distances as dist
import umato.sparse as sparse

from umato.utils import (
    adjacency_matrix,
    ts,
    csr_unique,
    plot_tmptmp,
)

from umato.layouts import (
    optimize_global_layout,
    nn_layout_optimize,
)

from umato.umap_ import (
    nearest_neighbors,
    fuzzy_simplicial_set,
    make_epochs_per_sample,
    find_ab_params,
)

try:
    # Use pynndescent, if installed (python 3 only)
    from pynndescent import NNDescent
    from pynndescent.distances import named_distances as pynn_named_distances
    from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances

    _HAVE_PYNNDESCENT = True
except ImportError:
    _HAVE_PYNNDESCENT = False


locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@numba.njit(
    # parallel=True,  # can SABOTAGE the array order (should be used with care)
    fastmath=True,
)
def build_knn_graph(
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


def build_global_structure(
    data,
    hubs,
    n_components,
    a,
    b,
    random_state,
    alpha=0.0065,
    n_epochs=30,
    verbose=Trues,
    label=None,
    init_global="pca",
):

    if init_global == "pca":
        Z = PCA(n_components=n_components).fit_transform(data[hubs])
        Z /= Z.max()
    elif init_global == "random":
        Z = np.random.random((len(hubs), n_components))
    else:
        raise ValueError("Check hub node initializing method!")

    P = adjacency_matrix(data[hubs])
    # P /= np.sum(P, axis=1, keepdims=True)
    P /= P.max()

    if verbose:
        result = optimize_global_layout(
            P=P,
            Z=Z,
            a=a,
            b=b,
            alpha=alpha,
            n_epochs=n_epochs,
            verbose=True,
            savefig=False,
            label=label[hubs],
        )
    else:
        result = optimize_global_layout(
            P, Z, a, b, alpha=alpha, n_epochs=n_epochs
        )  # (TODO) how to optimize n_epochs & alpha?

    return result


def embed_others_nn_progressive(
    data, init_global, original_hubs, hubs, knn_indices, nn_consider, random_state, label, last=False
):
    init = np.zeros((data.shape[0], init_global.shape[1]))
    init[hubs] = init_global

    if last:
        while True:
            val = len(hubs)
            hubs = hub_nn_num(
                data=data, hubs=hubs, knn_indices=knn_indices, nn_consider=nn_consider,
            )

            if val == len(hubs):
                if len(init) > len(hubs):
                    print(f"len(hubs) {len(hubs)} is smaller than len(init) {len(init)}")
                break

    else:
        hubs = hub_nn_num(
            data=data, hubs=hubs, knn_indices=knn_indices, nn_consider=nn_consider,
        )
        if len(init) > len(hubs):
            print(f"len(hubs) {len(hubs)} is smaller than len(init) {len(init)}")

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
        nn_consider=10, # number of hubs to consider
    )

    # np.array of hub information (hubs = 2, hub_nn = 1, outliers = 0)
    hub_info = np.zeros(data.shape[0])
    hub_info[hub_nn] = 1
    hub_info[original_hubs] = 2

    # save figure2
    plot_tmptmp(data=init[hubs], label=label[hubs], name=f"pic2")

    return init, hub_info, hubs


def embed_outliers(
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
    data, init, hubs, disjoints, random, nn_consider=1.0,
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

                nn_consider_tmp = nn_consider
                if len(distances) < nn_consider:
                    nn_consider_tmp = len(distances)

                ixs = np.array(distances).argsort()[:nn_consider_tmp]
                init[j] = np.zeros(init.shape[1])
                for ix in ixs:
                    target_ix = indices[ix]
                    init[j] += init[target_ix]
                init[j] /= nn_consider_tmp
                init[j] += random[j]  # add random value

                hubs.add(j)

    return init, hubs


@numba.njit()
def hub_nn_num(
    data, hubs, knn_indices, nn_consider=10,
):
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
    # locals={"dists": numba.types.float32[::1],},
    parallel=True,
    fastmath=True,
)
def apppend_knn(
    data, knn_indices, knn_dists, hub_info, n_neighbors, counts, counts_sum,
):
    for i in numba.prange(data.shape[0]):
        num = n_neighbors - counts[i]
        if hub_info[i] > 0 and num > 0:
            # found neighbors (# of neighbors < n_neighbors)
            neighbors = knn_indices[i][: counts[i]]

            # find unique target indices
            indices = set()
            for ci in range(counts[i]):  # cannot use numba.prange; malloc error occurs
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
    learning_rate,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    parallel=False,
    verbose=False,
    label=None,
    k=0,
):

    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    graph.data[
        hub_info[graph.col] == 2
    ] = 1.0  # current (NNs) -- other (hubs): 1.0 weight
    graph.data[
        hub_info[graph.row] == 2
    ] = 0.0  # current (hubs) -- other (hubs, nns): 0.0 weight (remove)
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

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

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

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
        gamma=gamma,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
        k=k,
        label=label,
    )

    return embedding


@numba.njit(
    locals={
        "out": numba.types.int32[:, ::1],
    },
    parallel=True,
    fastmath=True,
)
def change_indices(zz, hubs):
    out = np.zeros((zz.shape[0], zz.shape[1]), dtype=np.int32)
    for i in numba.prange(zz.shape[0]):
        for j in numba.prange(zz.shape[1]):
            out[i, j] = hubs[zz[i, j]]
    return out


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
        repulsion_strength=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        verbose=False,
        ll=None,
    ):
        self.n_neighbors = n_neighbors
        self.hub_num = hub_num
        self.metric = metric
        self.global_n_epochs = global_n_epochs
        self.local_n_epochs = local_n_epochs
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.global_learning_rate = global_learning_rate
        self.local_learning_rate = local_learning_rate
        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.verbose = verbose
        self.a = a
        self.b = b

        self.ll = ll

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self.global_learning_rate < 0.0 or self.local_learning_rate < 0.0:
            raise ValueError("learning_rates must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if not isinstance(self.hub_num, int) or self.hub_num < -1:
            raise ValueError("hub_num must be a positive integer or -1 (None)")
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
        if self.global_n_epochs is not None and (
            self.global_n_epochs > 0 or not isinstance(self.global_n_epochs, int)
        ):
            raise ValueError("global_n_epochs must be a positive integer")
        if self.local_n_epochs is not None and (
            self.local_n_epochs > 0 or not isinstance(self.local_n_epochs, int)
        ):
            raise ValueError("local_n_epochs must be a positive integer")

        # check sparsity of data
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False

        # set input distance metric
        if self.metric in dist.named_distances:
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
        else:
            raise ValueError("metric is not a recognised string")

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

    def fit(self, X):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)

        self._validate_parameters()

        if self.verbose:
            print(str(self))

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
            print(ts(), "Construct fuzzy simplicial set")

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
            self.angular_rp_forest,
            random_state,
            self.low_memory,
            use_pynndescent=True,
            verbose=self.verbose,
        )

        if self.local_n_epochs is None:
            self.local_n_epochs = 50

        if self.global_n_epochs is None:
            self.global_n_epochs = 100

        if self.verbose:
            print(ts(), "Build K-nearest neighbor graph structure")

        flat_indices = self._knn_indices.flatten()  # flattening all knn indices
        index, freq = np.unique(flat_indices, return_counts=True)
        # sorted_index = index[freq.argsort(kind="stable")]  # sorted index in increasing order
        sorted_index = index[
            freq.argsort(kind="stable")[::-1]
        ]  # sorted index in decreasing order

        # get disjoint NN matrix
        disjoints = build_knn_graph(
            data=X, sorted_index=sorted_index, hub_num=self.hub_num,
        )

        # get hub indices from disjoint set
        hubs = pick_hubs(disjoints=disjoints, random_state=random_state, popular=True,)

        if self.verbose:
            print(ts(), "Run global optimization")

        init_global = build_global_structure(
            data=X,
            hubs=hubs,
            n_components=self.n_components,
            a=self.a,
            b=self.b,
            random_state=random_state,
            alpha=self.global_learning_rate,
            n_epochs=self.global_n_epochs,
            verbose=True,
            label=self.ll,
        )

        if self.verbose:
            print(
                ts(), "Get NN indices & Initialize them using original hub information"
            )
        
        original_hubs = hubs.copy()

        # FIRST EMBEDDING
        init, hub_info, hubs = embed_others_nn_progressive(
            data=X,
            init_global=init_global,
            original_hubs=original_hubs,
            hubs=hubs,
            knn_indices=self._knn_indices,
            nn_consider=self._n_neighbors,
            random_state=random_state,
            label=self.ll,
            last=False,
        )

        (tmp_knn_indices, tmp_knn_dists, _) = nearest_neighbors(
            X[hubs],
            self._n_neighbors,
            nn_metric,
            self.angular_rp_forest,
            random_state,
            self.low_memory,
            use_pynndescent=True,
            verbose=self.verbose,
        )

        tmp_knn_indices = np.array(tmp_knn_indices, dtype=np.int32)
        tmp_knn_dists = np.array(tmp_knn_dists, dtype=np.int32)

        tmp_knn_indices = change_indices(tmp_knn_indices, hubs)
        tmp_knn_dists = change_indices(tmp_knn_dists, hubs)

        self.graph_, _, _ = fuzzy_simplicial_set(
            X[hubs],
            self.n_neighbors,
            random_state,
            nn_metric,
            hubs,
            tmp_knn_indices,
            tmp_knn_dists,
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
            True,
        )

        if self.verbose:
            print(ts(), "Run local optimization")

        init = local_optimize_nn(
            data=X,
            graph=self.graph_,
            hub_info=hub_info,
            n_components=self.n_components,
            learning_rate=self.local_learning_rate,
            a=self.a,
            b=self.b,
            gamma=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=10,
            init=init,
            random_state=random_state,
            parallel=False,
            verbose=True,
            k=0,
            label=self.ll,
        )

        print("finished 0")
        print(np.unique(hub_info, return_counts=True))

        # SECOND EMBEDDING
        init, hub_info, hubs = embed_others_nn_progressive(
            data=X,
            init_global=init[hubs],
            original_hubs=original_hubs,
            hubs=hubs,
            knn_indices=self._knn_indices,
            nn_consider=self._n_neighbors,
            random_state=random_state,
            label=self.ll,
            last=False,
        )

        (tmp_knn_indices, tmp_knn_dists, _) = nearest_neighbors(
            X[hubs],
            self._n_neighbors,
            nn_metric,
            self.angular_rp_forest,
            random_state,
            self.low_memory,
            use_pynndescent=True,
            verbose=self.verbose,
        )

        tmp_knn_indices = np.array(tmp_knn_indices, dtype=np.int32)
        tmp_knn_dists = np.array(tmp_knn_dists, dtype=np.int32)

        tmp_knn_indices = change_indices(tmp_knn_indices, hubs)
        tmp_knn_dists = change_indices(tmp_knn_dists, hubs)

        self.graph_, _, _ = fuzzy_simplicial_set(
            X[hubs],
            self.n_neighbors,
            random_state,
            nn_metric,
            hubs,
            tmp_knn_indices,
            tmp_knn_dists,
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
            True,
        )

        if self.verbose:
            print(ts(), "Run local optimization")

        init = local_optimize_nn(
            data=X,
            graph=self.graph_,
            hub_info=hub_info,
            n_components=self.n_components,
            learning_rate=self.local_learning_rate,
            a=self.a,
            b=self.b,
            gamma=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=10,
            init=init,
            random_state=random_state,
            parallel=False,
            verbose=True,
            k=1,
            label=self.ll,
        )

        print("finished 1")
        print(np.unique(hub_info, return_counts=True))

        # THIRD EMBEDDING
        init, hub_info, hubs = embed_others_nn_progressive(
            data=X,
            init_global=init[hubs],
            original_hubs=original_hubs,
            hubs=hubs,
            knn_indices=self._knn_indices,
            nn_consider=self._n_neighbors,
            random_state=random_state,
            label=self.ll,
            last=True,
        )

        (tmp_knn_indices, tmp_knn_dists, _) = nearest_neighbors(
            X[hubs],
            self._n_neighbors,
            nn_metric,
            self.angular_rp_forest,
            random_state,
            self.low_memory,
            use_pynndescent=True,
            verbose=self.verbose,
        )

        tmp_knn_indices = np.array(tmp_knn_indices, dtype=np.int32)
        tmp_knn_dists = np.array(tmp_knn_dists, dtype=np.int32)

        tmp_knn_indices = change_indices(tmp_knn_indices, hubs)
        tmp_knn_dists = change_indices(tmp_knn_dists, hubs)

        self.graph_, _, _ = fuzzy_simplicial_set(
            X[hubs],
            self.n_neighbors,
            random_state,
            nn_metric,
            hubs,
            tmp_knn_indices,
            tmp_knn_dists,
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
            True,
        )

        if self.verbose:
            print(ts(), "Run local optimization")

        init = local_optimize_nn(
            data=X,
            graph=self.graph_,
            hub_info=hub_info,
            n_components=self.n_components,
            learning_rate=self.local_learning_rate,
            a=self.a,
            b=self.b,
            gamma=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=40,
            init=init,
            random_state=random_state,
            parallel=False,
            verbose=True,
            k=2,
            label=self.ll,
        )

        print("finished 2")
        print(np.unique(hub_info, return_counts=True))

        if self.verbose:
            print(ts(), "Embedding outliers")

        self.embedding_ = embed_outliers(
            data=X,
            init=init,
            hubs=hubs,
            disjoints=disjoints,
            random_state=random_state,
            label=self.ll,
        )

        if self.verbose:
            print(ts(), "Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)

        return self


    def fit_transform(self, X):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X)
        return self.embedding_