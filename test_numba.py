
import numba
import numpy as np
from evaluation.models.dataset import get_data, save_csv
import time

@numba.njit() # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

@numba.njit()
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

@numba.njit()
def myfunc(X, hub_idx):
    hub_idx = set(hub_idx)  # use set for fast computation
    hub_not_idx = set(list(range(X.shape[0])))
    hub_not_idx -= hub_idx

    for _ in range(5):

        for i in hub_not_idx:
            # targets = []
            for j in hub_idx:
                if i in knn_indices[j]:
                    # targets.append(j)
                    hub_not_idx_fin -= {i}
                    hub_idx_fin.add(i)
                    break

        hub_idx = hub_idx_fin.copy()
        hub_not_idx = hub_not_idx_fin.copy()

    return 0

@numba.njit()
def myfunc2(hub_idx):
    x = np.arange(10)
    y = np.delete(x, hub_idx)
    return y


@numba.njit()
def z1():
    return 0

@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]

if __name__ == "__main__":
    INT32_MAX = np.iinfo(np.int32).max - 1
    INT32_MIN = np.iinfo(np.int32).min - 1
    random_state = np.random.mtrand._rand
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    
    print(rng_state)
    print(tau_rand_int(rng_state))