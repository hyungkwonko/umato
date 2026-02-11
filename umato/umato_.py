"""
Author: Hyung-Kwon Ko (hkko@hcil.snu.ac.kr)
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
from sklearn.manifold import SpectralEmbedding

from umato.utils import (
    adjacency_matrix,
    ts,
    csr_unique,
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
MAX_SPARSE_TO_DENSE_ELEMENTS = 50_000_000


@numba.njit(parallel=True, fastmath=True)
def calculate_distances(data, source, targets):
    distances_squared = np.zeros(len(targets), dtype=data.dtype)
    for target_idx in numba.prange(len(targets)):
        target = targets[target_idx]
        distance_squared = np.sum((data[source] - data[target]) ** 2)
        distances_squared[target_idx] = distance_squared
    return distances_squared


@numba.njit(
    # parallel=True,  # can SABOTAGE the array order (should be used with care)
    fastmath=True,
)
def build_knn_graph(
    data, sorted_index, hub_num,
):
    sorted_index_c = sorted_index.copy()

    index_map = [0] * data.shape[0]
    for i, val in enumerate(sorted_index_c):
        index_map[val] = i
    index_map = np.array(index_map)

    disjoints = []

    real_point_count = data.shape[0]
    # the number of nodes per KNN, i.e. the K value
    leaf_num = int(np.ceil(real_point_count / hub_num))

    # the number of points to be appended to the disjoints list
    total_disjoints_count = leaf_num * hub_num

    # the number of padding points, having index value -1
    # this number will be smaller than leaf_num, as empty KNNs won't be created
    padding_point_count = (total_disjoints_count - real_point_count) % leaf_num

    # the number of hubs with no padding points
    full_real_knn_hub_count = int(np.floor(real_point_count / leaf_num))

    last_source_loc = -1

    # deal with hubs with no padding points first
    for i in range(full_real_knn_hub_count):
        # append the first element
        for j in range(last_source_loc + 1, len(sorted_index_c)):
            if sorted_index_c[j] > -1:
                source = sorted_index_c[j]
                sorted_index_c[j] = -1
                last_source_loc = j
                break
        else:
            # No hub source found!
            raise AssertionError("Logic should not reach this block.")

        # get distance for each element
        targets = sorted_index[sorted_index_c != -1]
        distances_squared = calculate_distances(data, source, targets)

        # append other elements
        sorted_distance_indexes = targets[np.argsort(distances_squared)[:leaf_num - 1]].astype(np.int64)

        # create KNN
        disjoint = [source] + list(sorted_distance_indexes)

        # mark indexes as used
        sorted_index_c[index_map[sorted_distance_indexes]] = -1

        disjoints.append(disjoint)

    if padding_point_count:
        source = -1
        disjoint = []

        # append the first element
        for j in range(last_source_loc + 1, len(sorted_index_c)):
            if sorted_index_c[j] > -1:
                source = sorted_index_c[j]
                disjoint.append(source)
                sorted_index_c[j] = -1
                last_source_loc = j
                break
        if source == -1:
            # No hub source found!
            raise AssertionError("Logic should not reach this block.")

        # get distance for each element
        targets = sorted_index[sorted_index_c != -1]
        distances_squared = calculate_distances(data, source, targets)

        # append other elements
        real_elements = leaf_num - padding_point_count
        sorted_distance_indexes = targets[np.argsort(distances_squared)[:real_elements - 1]].astype(np.int64)

        # create KNN
        disjoint = [source] + list(sorted_distance_indexes) + [-1] * padding_point_count

        # mark indexes as used
        sorted_index_c[index_map[sorted_distance_indexes]] = -1

        disjoints.append(disjoint)

    return np.array(disjoints)


def build_knn_graph_from_knn(
    sorted_index,
    knn_indices,
    hub_num,
):
    n_samples = knn_indices.shape[0]
    leaf_num = int(np.ceil(n_samples / hub_num))
    n_groups = int(np.ceil(n_samples / leaf_num))

    remaining = np.ones(n_samples, dtype=np.bool_)
    order = sorted_index.astype(np.int64, copy=False)
    disjoints = []

    for _ in range(n_groups):
        source = -1
        for idx in order:
            if remaining[idx]:
                source = int(idx)
                break

        if source < 0:
            break

        remaining[source] = False
        disjoint = [source]

        queue = [source]
        seen = {source}
        q_ptr = 0

        while q_ptr < len(queue) and len(disjoint) < leaf_num:
            node = queue[q_ptr]
            q_ptr += 1

            for nb in knn_indices[node]:
                nb = int(nb)
                if nb < 0:
                    continue
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
                if remaining[nb]:
                    remaining[nb] = False
                    disjoint.append(nb)
                    if len(disjoint) >= leaf_num:
                        break

        if len(disjoint) < leaf_num:
            for idx in order:
                idx = int(idx)
                if remaining[idx]:
                    remaining[idx] = False
                    disjoint.append(idx)
                    if len(disjoint) >= leaf_num:
                        break

        if len(disjoint) < leaf_num:
            disjoint.extend([-1] * (leaf_num - len(disjoint)))

        disjoints.append(disjoint)

    return np.array(disjoints, dtype=np.int64)


def fill_missing_knn_from_source(
    selected_indices,
    selected_dists,
    source_indices,
    source_dists,
    counts,
    n_neighbors,
):
    remaining = 0
    for i in range(selected_indices.shape[0]):
        need = n_neighbors - int(counts[i])
        if need <= 0:
            continue

        present = set(selected_indices[i, : counts[i]].tolist())
        insert_pos = int(counts[i])

        for j in range(source_indices.shape[1]):
            cand = int(source_indices[i, j])
            if cand < 0 or cand in present:
                continue
            selected_indices[i, insert_pos] = cand
            selected_dists[i, insert_pos] = source_dists[i, j]
            present.add(cand)
            insert_pos += 1
            if insert_pos >= n_neighbors:
                break

        if insert_pos < n_neighbors:
            remaining += 1

    return selected_indices, selected_dists, remaining


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
            raise ValueError(f"hub_num({hub_num}) is not the same as hubs({hubs})!")

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
    verbose=False,
    label=None,
    init_global="pca",
):
    hub_data = data[hubs]
    if scipy.sparse.issparse(hub_data):
        hub_data = np.asarray(hub_data.toarray(), dtype=np.float32, order="C")
    else:
        hub_data = np.asarray(hub_data, dtype=np.float32, order="C")

    if isinstance(init_global, str):
        if init_global == "pca":
            Z = PCA(n_components=n_components).fit_transform(hub_data)
            Z /= Z.max()
        elif init_global == "random":
            Z = random_state.normal(
                loc=0.0, scale=0.05, size=list((len(hubs), n_components))
            ).astype(np.float32)
            Z /= Z.max()
        elif init_global == "spectral":
            Z = SpectralEmbedding(n_components=n_components).fit_transform(hub_data)
        else:
            raise ValueError("Check hub node initializing method!")
    else:
        Z = init_global[hubs]
        Z /= Z.max()

    P = adjacency_matrix(hub_data)
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
            verbose=verbose,
        )
    else:
        result = optimize_global_layout(
            P, Z, a, b, alpha=alpha, n_epochs=n_epochs
        )  # (TODO) how to optimize n_epochs & alpha?

    return result


def embed_others_nn(
    data, init_global, hubs, knn_indices, nn_consider, random_state, label, verbose=False,
):
    init = np.zeros((data.shape[0], init_global.shape[1]))
    original_hubs = hubs.copy()
    init[original_hubs] = init_global

    max_iterations = data.shape[0] + 1
    iterations = 0
    while True:
        val = len(hubs)
        hubs = hub_nn_num(
            data=data, hubs=hubs, knn_indices=knn_indices, nn_consider=nn_consider,
        )
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                "Exceeded hub expansion iteration guard while initializing neighbors."
            )

        if val == len(hubs):
            if len(init) > len(hubs) and verbose:
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
        nn_consider=10 if len(original_hubs) >= 10 else len(original_hubs), # number of hubs to consider
    )

    # np.array of hub information (hubs = 2, hub_nn = 1, outliers = 0)
    hub_info = np.zeros(data.shape[0])
    hub_info[hub_nn] = 1
    hub_info[original_hubs] = 2

    return init, hub_info, hubs


def embed_others_nn_from_graph(
    n_samples,
    init_global,
    hubs,
    knn_indices,
    knn_dists,
    nn_consider,
    random_state,
    verbose=False,
):
    init = np.zeros((n_samples, init_global.shape[1]), dtype=np.float32)
    original_hubs = np.asarray(hubs, dtype=np.int64)
    init[original_hubs] = init_global

    max_iterations = n_samples + 1
    iterations = 0
    hubs_expanded = original_hubs.copy()

    while True:
        prev_len = len(hubs_expanded)
        hubs_expanded = hub_nn_num(
            data=np.empty((n_samples, 1), dtype=np.float32),
            hubs=hubs_expanded,
            knn_indices=knn_indices,
            nn_consider=nn_consider,
        )
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                "Exceeded hub expansion iteration guard while initializing sparse neighbors."
            )
        if prev_len == len(hubs_expanded):
            if n_samples > len(hubs_expanded) and verbose:
                print(f"len(hubs) {len(hubs_expanded)} is smaller than n_samples {n_samples}")
            break

    hub_nn = np.array(list(set(hubs_expanded) - set(original_hubs)), dtype=np.int64)
    original_hub_set = set(int(x) for x in original_hubs.tolist())
    fallback_mean = init[original_hubs].mean(axis=0) if len(original_hubs) else np.zeros(init.shape[1], dtype=np.float32)

    for idx in hub_nn:
        neighbors = knn_indices[idx][:nn_consider]
        neighbor_dists = knn_dists[idx][:nn_consider]

        hub_candidates = []
        hub_weights = []
        for nb, dist_val in zip(neighbors, neighbor_dists):
            nb = int(nb)
            if nb in original_hub_set:
                hub_candidates.append(nb)
                hub_weights.append(1.0 / (float(dist_val) + 1e-6))

        if hub_candidates:
            w = np.asarray(hub_weights, dtype=np.float32)
            w /= w.sum()
            init[idx] = np.sum(init[np.asarray(hub_candidates, dtype=np.int64)] * w[:, None], axis=0)
        else:
            sample_size = min(10, len(original_hubs))
            sampled = random_state.choice(original_hubs, size=sample_size, replace=False)
            init[idx] = init[sampled].mean(axis=0) if sample_size > 0 else fallback_mean

        init[idx] += random_state.normal(
            loc=0.0, scale=0.05, size=init.shape[1]
        ).astype(np.float32)

    hub_info = np.zeros(n_samples)
    hub_info[hub_nn] = 1
    hub_info[original_hubs] = 2
    return init, hub_info, hubs_expanded


def embed_outliers(
    data, init, hubs, disjoints, random_state, label, n_neighbors, verbose=False,
):
    # generate random normal distribution
    random_normal = random_state.normal(scale=0.02, size=list(init.shape)).astype(
        np.float32
    )

    # append other nodes using NN disjoint information
    init, nodes_number = disjoint_initialize(
        data=data, init=init, hubs=hubs, disjoints=disjoints, random=random_normal, nn_consider=n_neighbors
    )

    if len(init) != len(nodes_number):
        raise ValueError(
            f"total data # ({len(init)}) != total embedded # ({len(nodes_number)})!"
        )

    return init


def embed_outliers_from_knn(
    init,
    hub_info,
    knn_indices,
    random_state,
    nn_consider,
):
    assigned = hub_info > 0
    noise_scale = 0.02

    max_passes = 10
    for _ in range(max_passes):
        changed = 0
        for i in range(init.shape[0]):
            if assigned[i]:
                continue

            neighbors = knn_indices[i][:nn_consider]
            valid = [int(nb) for nb in neighbors if nb >= 0 and assigned[int(nb)]]
            if valid:
                init[i] = init[np.asarray(valid, dtype=np.int64)].mean(axis=0)
                init[i] += random_state.normal(
                    loc=0.0, scale=noise_scale, size=init.shape[1]
                ).astype(np.float32)
                assigned[i] = True
                changed += 1

        if changed == 0:
            break

    unresolved = np.where(~assigned)[0]
    if unresolved.size > 0:
        if np.any(assigned):
            base = init[assigned].mean(axis=0)
        else:
            base = np.zeros(init.shape[1], dtype=np.float32)
        for i in unresolved:
            init[i] = base + random_state.normal(
                loc=0.0, scale=noise_scale, size=init.shape[1]
            ).astype(np.float32)

    return init


@numba.njit()
def disjoint_initialize(
    data, init, hubs, disjoints, random, nn_consider
):

    hubs_true = np.zeros(data.shape[0])
    hubs_true[hubs] = True
    hubs = set(hubs)

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
            for j in range(knn_indices.shape[1]):
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
    fastmath=True,
)
def apppend_knn(
    data, knn_indices, knn_dists, hub_info, n_neighbors, counts, counts_sum,
):
    for i in range(data.shape[0]):
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
                for j in range(num):
                    knn_indices[i][counts[i] + j] = targets[
                        sorted_dists_index[j]
                    ]
                    knn_dists[i][counts[i] + j] = dists[
                        sorted_dists_index[j]
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
):
    if negative_sample_rate <= 0:
        raise ValueError("negative_sample_rate must be greater than 0")
    if np.count_nonzero(hub_info > 0) == 0:
        raise ValueError("No hub-linked vertices available for negative sampling.")

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
        verbose=verbose
    )

    return embedding


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
        verbose=False,
    ):
        self.n_neighbors = n_neighbors
        self.hub_num = hub_num
        self.metric = metric
        self.global_n_epochs = global_n_epochs
        self.local_n_epochs = local_n_epochs
        self.n_components = n_components
        self.gamma = gamma
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
        self.init = init

        self.ll = ll

    def _validate_parameters(self):
        n_samples = self._raw_data.shape[0]

        if not isinstance(self.hub_num, int):
            raise ValueError("hub_num must be an integer")
        if self.hub_num < 2:
            raise ValueError("hub_num must be at least 2")
        if self.hub_num >= n_samples:
            raise ValueError("hub_num must be less than the number of data points")

        if not isinstance(self.n_neighbors, int):
            raise ValueError("n_neighbors must be an integer")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")

        if not isinstance(self.negative_sample_rate, (int, float)):
            raise ValueError("negative_sample_rate must be a numeric value")
        if self.negative_sample_rate <= 0:
            raise ValueError("negative_sample_rate must be greater than 0")

        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.gamma < 0.0:
            raise ValueError("gamma cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.global_learning_rate < 0.0 or self.local_learning_rate < 0.0:
            raise ValueError("learning_rates must be positive")
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

        if isinstance(self.init, np.ndarray):
            if self.init.ndim != 2:
                raise ValueError("init array must be 2D with shape (n_samples, n_components)")
            if self.init.shape[0] != n_samples:
                raise ValueError(
                    "init array first dimension must match n_samples "
                    f"({self.init.shape[0]} != {n_samples})"
                )
            if self.init.shape[1] != self.n_components:
                raise ValueError(
                    "init array second dimension must match n_components "
                    f"({self.init.shape[1]} != {self.n_components})"
                )
        elif not isinstance(self.init, str):
            raise ValueError('init must be one of {"pca", "random", "spectral"} or a numpy.ndarray')

        if self.global_n_epochs is not None and (
            not isinstance(self.global_n_epochs, int) or self.global_n_epochs < 10
        ):
            raise ValueError("global_n_epochs must be a positive integer of at least 10")
        if self.local_n_epochs is not None and (
            not isinstance(self.local_n_epochs, int) or self.local_n_epochs < 10
        ):
            raise ValueError("local_n_epochs must be a positive integer of at least 10")

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
        elif self.metric == 'precomputed':
            self._input_distance_func = 'precomputed'
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

        X_checked = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._sparse_native_mode = False
        if scipy.sparse.isspmatrix_csr(X_checked):
            total_elements = X_checked.shape[0] * X_checked.shape[1]
            if total_elements > MAX_SPARSE_TO_DENSE_ELEMENTS:
                X = X_checked
                self._sparse_native_mode = True
            else:
                X = np.asarray(X_checked.toarray(), dtype=np.float32, order="C")
        else:
            X = np.asarray(X_checked, dtype=np.float32, order="C")

        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)

        self._validate_parameters()

        if self.verbose:
            print(str(self))

        # Error check n_neighbors based on data size
        if X.shape[0] <= self.n_neighbors:
            if X.shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

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
            X,
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
        missing_indices = np.setdiff1d(
            np.arange(X.shape[0], dtype=np.int64),
            sorted_index.astype(np.int64),
            assume_unique=False,
        )
        if missing_indices.size > 0:
            sorted_index = np.concatenate([sorted_index, missing_indices]).astype(
                np.int64, copy=False
            )

        # get disjoint NN matrix
        if self._sparse_data:
            disjoints = build_knn_graph_from_knn(
                sorted_index=sorted_index,
                knn_indices=self._knn_indices,
                hub_num=self.hub_num,
            )
        else:
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
            verbose=self.verbose,
            label=self.ll,
            init_global=self.init,
        )

        if self.verbose:
            print(
                ts(), "Get NN indices & Initialize them using original hub information"
            )

        if self._sparse_data:
            init, hub_info, hubs = embed_others_nn_from_graph(
                n_samples=X.shape[0],
                init_global=init_global,
                hubs=hubs,
                knn_indices=self._knn_indices,
                knn_dists=self._knn_dists,
                nn_consider=self._n_neighbors,
                random_state=random_state,
                verbose=self.verbose,
            )
        else:
            init, hub_info, hubs = embed_others_nn(
                data=X,
                init_global=init_global,
                hubs=hubs,
                knn_indices=self._knn_indices,
                nn_consider=self._n_neighbors,
                random_state=random_state,
                label=self.ll,
                verbose=self.verbose,
            )

        source_knn_indices = self._knn_indices
        source_knn_dists = self._knn_dists

        self._knn_indices, self._knn_dists, counts = select_from_knn(
            knn_indices=source_knn_indices,
            knn_dists=source_knn_dists,
            hub_info=hub_info,
            n_neighbors=self._n_neighbors,
            n=X.shape[0],
        )

        counts_hub = counts[hubs]
        counts_sum = len(counts_hub[counts_hub < self._n_neighbors])
        if counts_sum > 0:
            if self.verbose:
                print(ts(), "Adding more KNNs to build the graph")

            if self._sparse_data:
                self._knn_indices, self._knn_dists, counts_sum = fill_missing_knn_from_source(
                    selected_indices=self._knn_indices,
                    selected_dists=self._knn_dists,
                    source_indices=source_knn_indices,
                    source_dists=source_knn_dists,
                    counts=counts,
                    n_neighbors=self._n_neighbors,
                )
            else:
                self._knn_indices, self._knn_dists, counts_sum = apppend_knn(
                    data=X,
                    knn_indices=self._knn_indices,
                    knn_dists=self._knn_dists,
                    hub_info=hub_info,
                    n_neighbors=self._n_neighbors,
                    counts=counts,
                    counts_sum=counts_sum,
                )

            if counts_sum != 0:
                raise ValueError(
                    f"KNN indices not fully determined! counts_sum: {counts_sum} != 0"
                )

        self.graph_, _, _ = fuzzy_simplicial_set(
            X[hubs],
            self._n_neighbors,
            random_state,
            nn_metric,
            hubs,
            self._knn_indices[hubs],
            self._knn_dists[hubs],
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
            gamma=self.gamma,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=self.local_n_epochs,
            init=init,
            random_state=random_state,
            parallel=False,
            verbose=self.verbose,
            label=self.ll,
        )

        if self.verbose:
            print(ts(), "Embedding outliers")

        if self._sparse_data:
            self.embedding_ = embed_outliers_from_knn(
                init=init,
                hub_info=hub_info,
                knn_indices=source_knn_indices,
                random_state=random_state,
                nn_consider=self._n_neighbors,
            )
        else:
            self.embedding_ = embed_outliers(
                data=X,
                init=init,
                hubs=hubs,
                disjoints=disjoints,
                random_state=random_state,
                label=self.ll,
                n_neighbors=self._n_neighbors,
                verbose=self.verbose,
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
