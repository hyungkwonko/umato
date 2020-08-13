import numpy as np
import numba
import umato.distances as dist
from umato.utils import tau_rand_int
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


@numba.njit()
def clip(val, cutoff):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > cutoff:
        return cutoff
    elif val < -cutoff:
        return -cutoff
    else:
        return val


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.int32,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel
    )
    for n in range(n_epochs):
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding




################
# hyung-kwon ko
# hyung-kwon ko
# hyung-kwon ko

def get_CE(P, Y, d_squared, a, b):
    Q = pow(1 + a * d_squared ** b, -1)
    loss = -P * np.log(Q + 0.001) - (1 - P) * np.log(1 - Q + 0.001)
    return loss.sum() / 1e5


def get_DTM(adj_x, adj_z, sigma=0.1):
    density_x = calc_DTM(adj_x, sigma)
    density_z = calc_DTM(adj_z, sigma)
    return np.abs(density_x - density_z).sum()


def calc_DTM(adj, sigma):
    density = np.sum(np.exp(-(adj ** 2) / sigma), axis=-1)
    return density / density.sum()


def optimize_global_layout(
    P,
    Z,
    a,
    b,
    alpha=0.01,
    max_iter=10,
    verbose=False,
    savefig=False,
    label=None
):

    costs = []

    for i in range(max_iter):
        d_squared = np.square(euclidean_distances(Z))
        z_diff = np.expand_dims(Z, axis=1) - np.expand_dims(Z, axis=0)
        d_inverse = np.expand_dims(pow(1 + a * d_squared ** b, -1), axis=2)

        # Q is the normalized distance in low dimensional space 
        Q = pow(0.001 + d_squared, -1)
        np.fill_diagonal(Q, 0)
        Q = np.dot(1 - P, Q)
        Q /= np.sum(Q, axis=1, keepdims=True)
        # Q /= Q.max()

        grad = np.expand_dims(
            2 * a * b * P * (1e-12 + d_squared) ** (b - 1) - 2 * b * Q, axis=2
        )
        dZ = np.sum(grad * z_diff * d_inverse, axis=1)
        Z -= alpha * dZ

        if verbose:
            # cost = get_CE(P, Z, d_squared, a, b)
            cost = get_DTM(P, Q, sigma=0.1)
            costs.append(cost)
            print(
                f"[INFO] Current loss: {cost:.6f}, @ iteration: {i+1}/{max_iter}, alpha: {alpha}"
            )

        if savefig:
            if i % 4 == 1:
                from umato.umato_ import plot_tmptmp
                plot_tmptmp(data=Z, label=label, name=f"pic1_global{i}")

    return Z

# def optimize_global_layout(
#     P,
#     Z,
#     a,
#     b,
#     alpha=0.01,
#     max_iter=10,
#     verbose=False,
#     savefig=False,
#     label=None
# ):

#     costs = []

#     for i in range(max_iter):
#         d_squared = np.square(euclidean_distances(Z))
#         z_diff = np.expand_dims(Z, axis=1) - np.expand_dims(Z, axis=0)
#         d_inverse = np.expand_dims(pow(1 + a * d_squared ** b, -1), axis=2)

#         # Q is the normalized distance in low dimensional space 
#         Q = np.dot(1 - P, pow(0.001 + d_squared, -1))
#         np.fill_diagonal(Q, 0)
#         Q /= np.sum(Q, axis=1, keepdims=True)
#         # Q /= Q.max()

#         grad = np.expand_dims(
#             2 * a * b * P * (1e-12 + d_squared) ** (b - 1) - 2 * b * Q, axis=2
#         )
#         dZ = np.sum(grad * z_diff * d_inverse, axis=1)
#         Z -= alpha * dZ

#         if verbose:
#             # cost = get_CE(P, Z, d_squared, a, b)
#             cost = get_DTM(P, Q, sigma=1.0)
#             costs.append(cost)
#             print(
#                 f"[INFO] Current loss: {cost:.6f}, @ iteration: {i+1}/{max_iter}, alpha: {alpha}"
#             )

#         if savefig:
#             if i % 2 == 1:
#                 from umato.umato_ import plot_tmptmp
#                 plot_tmptmp(data=Z, label=label, name=f"pic1_global{i}")

#     return Z


def shaking(Z, num=-1):
    if num < 0:
        num = Z.shape[0] // 10

    centre = Z.mean(axis=0)
    distances = []
    for i in range(Z.shape[0]):
        distance = 0.0
        for d in range(Z.shape[1]):
            distance += (Z[i][d] - centre[d]) ** 2
        distances.append(distance)

    distances = np.array(distances)

    indices = np.argsort(distances)[-num:]
    for j in indices:
        Z[j] = centre + np.random.random(2) * 0.1

    return Z


def shaking2(Z, cutoff, times=1.25):
    centre = Z.mean(axis=0)
    for i in range(Z.shape[0]):
        distance = 0.0
        for d in range(Z.shape[1]):
            distance += (Z[i][d] - centre[d]) ** 2
        distance = np.sqrt(distance)
        if distance > (cutoff * times):
            Z[i] = (
                ((Z[i] - centre) / 2.0 / distance * cutoff)
                + np.random.random(2) * 0.5
                + centre
            )
    return Z


def get_max_hub(Z):
    centre = Z.mean(axis=0)
    cutoff = 0.0
    for i in range(Z.shape[0]):
        distance = 0.0
        for d in range(Z.shape[1]):
            distance += (Z[i][d] - centre[d]) ** 2
        if distance > cutoff:
            cutoff = distance
    return np.sqrt(cutoff)


def nn_layout_optimize(
    head_embedding,
    tail_embedding,
    head,
    tail,
    hub_info,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    label=None,
):

    (num, dim) = head_embedding.shape
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _nn_layout_optimize_single_epoch, fastmath=True, parallel=parallel
    )
    for n in range(n_epochs):
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            hub_info,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % 10 == 0:
            from umato.umato_ import plot_tmptmp

            plot_tmptmp(data=head_embedding, label=label, name=f"pic3_local{n}")
            # plot_tmptmp(data=tail_embedding, label=label, name=f"pic3_tail{n}")

        if verbose and n % 5 == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    plot_tmptmp(data=head_embedding, label=label, name=f"pic3_local{n}")
    return head_embedding


# def _nn_layout_optimize_single_epoch(
#     head_embedding,
#     tail_embedding,
#     head,
#     tail,
#     hub_info,
#     n_vertices,
#     epochs_per_sample,
#     a,
#     b,
#     rng_state,
#     gamma,
#     dim,
#     move_other,
#     alpha,
#     epochs_per_negative_sample,
#     epoch_of_next_negative_sample,
#     epoch_of_next_sample,
#     n,
# ):
#     for i in numba.prange(epochs_per_sample.shape[0]):
#         if epoch_of_next_sample[i] <= n:
#             j = head[i]  # j == source index
#             k = tail[i]  # k == target index

#             current = head_embedding[j]  # current == source location
#             other = tail_embedding[k]  # other == target location

#             dist_squared = rdist(current, other)  # get distance between them

#             if dist_squared > 0.0:
#                 grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
#                 grad_coeff /= a * pow(dist_squared, b) + 1.0
#             else:
#                 grad_coeff = 0.0

#             for d in range(dim):
#                 grad_d = grad_coeff * (current[d] - other[d])

#                 current[d] += grad_d * alpha

#                 grad_other = 1.0
#                 if hub_info[k] == 2:
#                     grad_other = 0.01

#                 if move_other:
#                     other[d] += -grad_d * alpha * grad_other

#             epoch_of_next_sample[i] += epochs_per_sample[i]

#             n_neg_samples = int(
#                 (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
#             )

#             for p in range(n_neg_samples):
#                 while True:
#                     k = tau_rand_int(rng_state) % n_vertices
#                     if hub_info[k] > 0:
#                         break

#                 other = tail_embedding[k]
#                 dist_squared = rdist(current, other)

#                 if dist_squared > 0.0:
#                     grad_coeff = 2.0 * gamma * b
#                     grad_coeff /= (0.001 + dist_squared) * (
#                         a * pow(dist_squared, b) + 1
#                     )
#                 elif j == k:
#                     continue
#                 else:
#                     grad_coeff = 0.0

#                 for d in range(dim):
#                     if grad_coeff > 0.0:
#                         grad_d = grad_coeff * (current[d] - other[d])
#                     else:
#                         grad_d = 0.0

#                     current[d] += grad_d * alpha

#             epoch_of_next_negative_sample[i] += (
#                 n_neg_samples * epochs_per_negative_sample[i]
#             )



def _nn_layout_optimize_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    hub_info,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]  # j == source index
            k = tail[i]  # k == target index

            current = head_embedding[j]  # current == source location
            other = tail_embedding[k]  # other == target location

            dist_squared = rdist(current, other)  # get distance between them

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]), 10.0)

                grad_other = 0.0
                grad_current = 0.0
                grad_neg = 0.001
                if hub_info[k] == 1:
                    grad_current = 0.005
                    grad_other = 0.005
                elif hub_info[k] == 2:
                    grad_current = 0.005
                    grad_other = 0.001

                # grad_other = 0.0
                # grad_current = 0.0
                # grad_neg = 0.05
                # if hub_info[k] == 1:
                #     grad_current = 0.05
                #     grad_other = 0.05
                # elif hub_info[k] == 2:
                #     grad_current = 0.05
                #     grad_other = 0.05

                current[d] += grad_d * alpha * grad_current

                if move_other:
                    other[d] += -grad_d * alpha * grad_other

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                while True:
                    k = tau_rand_int(rng_state) % n_vertices
                    if hub_info[k] > 0:
                        break

                other = tail_embedding[k]
                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0


                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]), 10.0)
                    else:
                        grad_d = 10.0

                    current[d] += grad_d * alpha * grad_neg

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
