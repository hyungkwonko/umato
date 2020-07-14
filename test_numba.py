
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


if __name__ == "__main__":
    X, label = get_data("spheres")  # spheres, mnist, fmnist, cifar10
    X = X[:1000]

    # go_fast(X)
    # sum2d(X)
    arr = np.array([1,2,3])
    # myfunc(X, arr)
    myfunc2(np.array([1,2]))
    t1 = time.time()
    
    myfunc2(np.array([1,2]))

    t2 = time.time()

    print(t2-t1)