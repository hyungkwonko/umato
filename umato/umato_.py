# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
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
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba

import umato.distances as dist

import umato.sparse as sparse
import umato.sparse_nndescent as sparse_nn

from umato.utils import (
    adjacency_matrix,
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
from umato.layouts import (
    optimize_global_layout,
    nn_layout_optimize,
)

from umato.umap_utils import (
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

import gudhi as gd

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


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
    dist = adjacency_matrix(data)
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
    hub_num=100,
    iter_num=5,
    interval=25,
    top_num=15,
    cutoff=0.05,
    local_knum=7,
):
    print("[INFO]: Select hub nodes using homology")

    hubs_list2 = None

    while True:

        if hubs_list2 == None:
            hubs_list, disjoints = hub_candidates(
                data, sorted_index, random_state, hub_num, iter_num
            )

            k1_list = []
            local_knum = int(hub_num * 0.2)
            for i in range(iter_num):
                d1 = data[hubs_list[i]]
                k1 = get_homology(d1, local_knum, top_num, random_state)
                k1_list.append(k1)
        else:
            hubs_list = hubs_list2.copy()
            disjoints = disjoints2.copy()
            k1_list = k2_list.copy()

        hubs_list2, disjoints2 = hub_candidates(
            data, sorted_index, random_state, hub_num + interval, iter_num
        )

        k2_list = []
        local_knum = int((hub_num + interval) * 0.2)
        for i in range(iter_num):
            d2 = data[hubs_list2[i]]
            k2 = get_homology(d2, local_knum, top_num, random_state)
            k2_list.append(k2)

        results = []
        for _k1 in k1_list:
            for _k2 in k2_list:
                # result = gd.bottleneck_distance(_k1, _k2, 0.01)  # approximation
                result = gd.bottleneck_distance(_k1, _k2)
                results.append(result)

        val = np.mean(results)
        # print(val)
        print(f"hub_num: {hub_num}, val: {val}")

        hub_num += interval

        if hub_num > 450:
            break

        # if val < cutoff:
        #     break
        # elif hub_num > 300:  # break if > 300
        #     warn(f"Hub node number set to {hub_num}!")
        #     break
        # else:
        #     hub_num += interval

    hubs = pick_hubs(disjoints=disjoints, random_state=random_state, popular=True,)
    exit()

    return hub_num



# def hub_candidates(
#     n, random_state, hub_num, iter_num=5,
# ):
#     hubs_list = []
#     indices = np.arange(n)
#     for i in range(iter_num):
#         hubs = random_state.choice(indices, hub_num, replace=False)
#         hubs_list.append(hubs)

#     return hubs_list


# def get_homology(data, local_knum, top_num, random_state, max_dimension=1, max_edge_length=1.0, num_samples=10, metric=gd.bottleneck_distance, level=0.95):
#     dist = adjacency_matrix(data)
#     dist /= dist.max()

#     # kill using local connectivity
#     nn_index = np.argpartition(dist, kth=local_knum - 1, axis=-1)[:,:local_knum]
#     for i in range(len(dist)):
#         dist[i][nn_index[i]] = random_state.random(local_knum) * 0.1

#     rc = gd.RipsComplex(distance_matrix=dist, max_edge_length=max_edge_length)
#     rc_tree = rc.create_simplex_tree(max_dimension=max_dimension+1)
#     rc_tree.persistence()
#     barcodes_list = [rc_tree.persistence_intervals_in_dimension(dim) for dim in np.arange(max_dimension+1)]

#     (n, _) = dist.shape

#     # Bottleneck bootstrap method for confidence sets of persistence diagrams for data filtrations built on data points
#     dist_vec = []
#     for _ in range(num_samples):
#         b = random_state.choice(n, n)
#         rc_b  = gd.RipsComplex(distance_matrix=data[b, :][:, b], max_edge_length=max_edge_length)
#         rc_tree_b = rc_b.create_simplex_tree(max_dimension=max_dimension+1)
#         rc_tree_b.persistence()

#         bot_b = 0
#         for dim in np.arange(max_dimension + 1):
#             interv_b_dim =  rc_tree_b.persistence_intervals_in_dimension(dim)
#             bot_b  = max(bot_b, metric(barcodes_list[dim], interv_b_dim))

#         dist_vec.append(bot_b)

#     quantile = np.quantile(dist_vec, level)

#     # calculate cutoff value
#     cutoff = quantile * 2

#     # sort out important topological features from noises
#     hom0 = rc_tree.persistence_intervals_in_dimension(0)
#     hom0 = hom0[np.where(abs(hom0[:,0] - hom0[:,1]) > cutoff)]
#     hom1 = rc_tree.persistence_intervals_in_dimension(1)
#     hom1 = hom1[np.where(abs(hom1[:,0] - hom1[:,1]) > cutoff)]

#     print(barcodes_list)
#     print(quantile)
#     print(hom0)
#     print(hom1)
#     exit()

#     # # we use top n topological features
#     # hom1_max = abs(hom1[:, 1] - hom1[:, 0])
#     # hom1_max_ix = hom1_max.argsort()[-top_num:][::-1]
#     # return hom1[hom1_max_ix]

#     return hom1


# def select_hubs_homology(
#     data,
#     random_state,
#     sorted_index,
#     hub_num=100,
#     hub_num_limit=500,
#     iter_num=5,
#     interval=10,
#     top_num=5,
#     cutoff=0.05,
#     local_knum=7,
# ):
#     print("[INFO]: Select hub nodes using homology")

#     # disjoints = disjoint_nn(data=data, sorted_index=sorted_index, hub_num=hub_num_limit,)

#     hubs_list2 = None

#     while True:

#         k1_list = []
#         if hub_num > 100:
#             hubs_list = hubs_list2.copy()
#             k1_list = k2_list.copy()
#         else:
#             hubs_list = hub_candidates(
#                 len(data), random_state, hub_num, iter_num
#             )
#             local_knum = int(hub_num * 0.1)
#             for i in range(iter_num):
#                 d1 = data[hubs_list[i]]
#                 k1 = get_homology(d1, local_knum, top_num, random_state)
#                 k1_list.append(k1)

#         hubs_list2 = hub_candidates(
#             len(data), random_state, hub_num + interval, iter_num
#         )

#         k2_list = []
#         for i in range(iter_num):
#             d2 = data[hubs_list2[i]]
#             local_knum = int((hub_num + interval) * 0.1)
#             k2 = get_homology(d2, local_knum, top_num, random_state)
#             k2_list.append(k2)

#         results = []
#         for _k1 in k1_list:
#             for _k2 in k2_list:
#                 # result = gd.bottleneck_distance(_k1, _k2, 0.01)  # approximation
#                 result = gd.bottleneck_distance(_k1, _k2)
#                 results.append(result)

#         val = np.mean(results)
#         print(f"hub_num: {hub_num}, val: {val}")

#         # hub_num += interval
#         if val < cutoff:
#             break
#         elif hub_num > hub_num_limit:
#             warn(f"Hub node number set to {hub_num}!")
#             break
#         else:
#             hub_num += interval

#     exit()

#     return hub_num



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
    alpha=0.006,
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

    P = adjacency_matrix(data[hubs])
    # P /= np.sum(P, axis=1, keepdims=True)
    P /= P.max()

    # local connectivity for global optimization
    # P = remove_local_connect(P, random_state)

    # import pandas as pd
    # import os
    # df = pd.DataFrame(Z)
    # df['label'] = label[hubs]
    # df.to_csv(os.path.join('global_init.csv'), index=False)
    # print(hubs[:100])
    # print("global init saved")


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

    # df = pd.DataFrame(result)
    # df['label'] = label[hubs]
    # df.to_csv(os.path.join('global_opt.csv'), index=False)
    # print("global init saved")
    # print(hubs[:100])

    return result


def embed_others_nn(
    data, init_global, hubs, knn_indices, nn_consider, random_state, label,
):
    init = np.zeros((data.shape[0], init_global.shape[1]))
    original_hubs = hubs.copy()
    init[original_hubs] = init_global

    print("[INFO] get hub_nn indices")

    while True:
        val = len(hubs)
        hubs = hub_nn_num(
            data=data, hubs=hubs, knn_indices=knn_indices, nn_consider=nn_consider,
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
            neighbors = knn_indices[i][:counts[i]]

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
        n_epochs = 50

    graph.data[hub_info[graph.col] == 2] = 1.0  # current (NNs) -- other (hubs): 1.0 weight
    graph.data[hub_info[graph.row] == 2] = 0.0  # current (hubs) -- other (hubs, nns): 0.0 weight (remove)
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    # graph.data[graph.data < 0.2] = 0.0
    graph.eliminate_zeros()

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

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    # import pandas as pd
    # import os
    # hubs = np.where(hub_info > 0)[0]
    # print(hubs[:100])
    # df = pd.DataFrame(embedding[hubs])
    # df['label'] = label[hubs]
    # df.to_csv(os.path.join('local_init.csv'), index=False)
    # print("local init saved")

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
        initial_alpha=initial_alpha,
        negative_sample_rate=negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
        label=label,
    )

    return embedding


class UMATO(BaseEstimator):
    def __init__(
        self,
        n_neighbors=50,
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
            self.hub_num = select_hubs_homology(
                data=X,
                random_state=random_state,
                sorted_index=sorted_index,
                hub_num=100,
            )

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
            alpha=0.0065,
            max_iter=150,
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
            nn_consider=self._n_neighbors,
            random_state=random_state,
            label=self.ll,
        )

        # exit()

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

        # with open("./hubs.npy", "wb") as f:
        #     np.save(f, hubs)

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

        # exit()

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
