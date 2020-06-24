"""
Utils

This includes quantitative measures to compare how the high-dimensional data and its low embedding result are similar.

Provides
  1. Root Mean Squared Error (RMSE)
  2. Mean Relative Rank Error (MRRE)
  3. Trustworthiness
  4. Continuity
  5. KL Divergence

References
----------
.. [1] Moor, M. et al. "Topological autoencoders." International
  Conference on Machine Learning (ICML). 2020.
  (preprint: https://arxiv.org/abs/1906.00722)

.. [2] original source code: https://osf.io/abuce/?view_only=f16d65d3f73e4918ad07cdd08a1a0d4b
"""

import numpy as np
from scipy.spatial.distance import pdist


def rmse(x, z):
    p_x = pdist(x)  # get pair-wise distance in high dimensional space
    p_z = pdist(z)  # get pair-wise distance in low dimensional space
    n = x.shape[0]
    sum_of_squared_differences = np.square(p_x - p_z).sum()  # sum(y - y hat)
    return np.sqrt((2 * sum_of_squared_differences) / n ** 2)
