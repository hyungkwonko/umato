"""
Auhtor: Hyung-Kwon Ko (hyungkwonko@gmail.com)
Kruskal's minimum spanning tree w/ union find
"""


# takes 28.7 seconds for spheres dataset
# class UnionFind():
#     def __init__(self, n):
#         self.parent_node = np.arange(n)  # much faster than list(range(n))
#         self.n = n

#     def union(self, id1, id2):  # merge two sets
#         if self.parent_node[id1] == self.parent_node[id2]:
#             return False
#         elif self.parent_node[id1] < self.parent_node[id2]:  # parent setting using lower value
#             s2 = self.set_find(id2)
#             self.parent_node[s2] = self.parent_node[id1]
#             return True
#         else:
#             s1 = self.set_find(id1)
#             self.parent_node[s1] = self.parent_node[id2]
#             return True

#     def set_find(self, id):  # find set having the same parent id
#         return [i for i, e in enumerate(self.parent_node) if e == self.parent_node[id]]


# takes forever for spheres dataset
# class UnionFind():
#     def __init__(self, n):
#         self.parent_node = list(range(n))
#         self.n = n

#     def union(self, id1, id2):  # merge two sets
#         if self.parent_node[id1] == self.parent_node[id2]:
#             return False
#         elif self.parent_node[id1] < self.parent_node[id2]:  # parent setting using lower value
#             s2 = set_find(self.parent_node, id2)
#             for s in s2:
#                 self.parent_node[s] = self.parent_node[id1]
#             return True
#         else:
#             s1 = set_find(self.parent_node, id1)
#             for s in s1:
#                 self.parent_node[s] = self.parent_node[id2]
#             return True

# @numba.jit(nopython=True, parallel=True)
# def set_find(parent_node, id):  # find set having the same parent id
#     ret = []
#     for (i, e) in enumerate(parent_node):
#         if e == parent_node[id]:
#             ret.append(i)
#     return ret