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



def build_mst(data, graph):

    """
    1. build connected components with topological distance using coordinate format matrix (graph)
    2. check if it is a GCC
    3-1. if GCC, then return it as mst
    3-2. else get the minimum topological distance between each components (TODO)
    4. bridge all the connected components using the min topological distance btw connected components (TODO)
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    graph.eliminate_zeros()

    n = data.shape[0]  # number of data
    z =0
    for k in range(data.shape[1]):
        z += data[:,k].std()
    print(data.std())
    print(z / data.shape[1])
    data /= data.max()
    z = 0
    for k in range(data.shape[1]):
        z += data[:,k].std()
    print(data.std())
    print(z / data.shape[1])
    exit()

    components = list()  # list of connected components
    index_list = set(range(n))

    print(len(graph.data))
    # cutoff = np.mean(graph.data)
    # print(cutoff)
    hist = np.histogram(graph.data,bins=np.linspace(0,1,num=101))
    print(hist)
    cutoff = 0.75  # 587 components
    cutoff = hist[0][hist[1].index(max(hist[1]))]
    print(cutoff)
    ixs = [i for i in range(len(graph.data)) if graph.data[i] < cutoff]  ############### how to set cutoff????
    print(len(ixs))
    graph.data[ixs] = -1
    graph.row[ixs] = -1
    graph.col[ixs] = -1

    while len(index_list) > 0:
        index = set({index_list.pop()})  # choose one from a set
        for i in range(len(graph.data)):
            if (graph.row[i] in index) or (graph.col[i] in index):
                target = {graph.row[i], graph.col[i]}
                index.update(target)
                index_list -= target  # remove from index list
        components.append(index)  # append connected components

    ######### need to make edge weight symmetric in each component ############
    ######### need to make edge weight symmetric in each component ############
    ######### need to make edge weight symmetric in each component ############
    ######### need to make edge weight symmetric in each component ############

    if len(components) == 1:
        print("[INFO] This is a minimum spanning tree.")
    else:
        print(f"[INFO] This is not a MST w/ {len(components)} components")
        exit()  # (TODO) implementation of 3-2 & 4 required

    # Build MST using Kruskal's algorithm w/ union finder
    edge_val_set = dict()
    indices = np.argsort(
        -graph.data
    )  # sorted indices in descending order(-) / ascending order(+)
    uf = UnionFind(n)

    zzz = []  #############

    for j in range(len(graph.data)):
        ix = indices[j]
        no_cycle = uf.union(graph.col[ix], graph.row[ix])
        if no_cycle:
            if graph.col[ix] not in [*edge_val_set]:
                edge_val_set[graph.col[ix]] = [[], []]
            edge_val_set[graph.col[ix]][0].append(graph.row[ix])
            edge_val_set[graph.col[ix]][1].append(graph.data[ix])
            if graph.row[ix] not in [*edge_val_set]:
                edge_val_set[graph.row[ix]] = [[], []]
            edge_val_set[graph.row[ix]][0].append(graph.col[ix])
            edge_val_set[graph.row[ix]][1].append(graph.data[ix])
            zzz.append(graph.data[ix])  #############


    print(len([l for l in zzz if l < 0.85]))  #############
    print(len([l for l in zzz if l < 0.86]))  #############
    print(len([l for l in zzz if l < 0.87]))  #############
    print(len([l for l in zzz if l < 0.88]))  #############
    print(len([l for l in zzz if l < 0.89]))  #############
    print(len([l for l in zzz if l < 0.90]))  #############
    m = sum(zzz) / len(zzz)
    print(m)
    print(len([l for l in zzz if l < m]))  #############
    print(len(zzz))
    exit()  #############

    # for double checking
    len_edge_val_set = 0
    for k in edge_val_set.keys():
        len_edge_val_set += len(edge_val_set[k][0])
    len_edge_val_set = len_edge_val_set / 2 + 1

    if len_edge_val_set != n:
        raise ValueError(
            f"[ERROR] number of edges +1 ({len_edge_val_set}) should match the total node number ({n})!"
        )

    return edge_val_set


# takes 11.5 seconds for spheres dataset
class UnionFind:
    def __init__(self, n):
        self.parent_node = list(range(n))
        self.n = n

    def union(self, id1, id2):  # merge two sets
        if self.parent_node[id1] == self.parent_node[id2]:
            return False
        elif (
            self.parent_node[id1] < self.parent_node[id2]
        ):  # parent setting using lower value
            s2 = self.set_find(id2)
            for s in s2:
                self.parent_node[s] = self.parent_node[id1]
            return True
        else:
            s1 = self.set_find(id1)
            for s in s1:
                self.parent_node[s] = self.parent_node[id2]
            return True

    # THIS takes most of the time (more than 95 %)
    def set_find(self, id):  # find set having the same parent id
        s = set({id})
        for i in range(self.n):
            if self.parent_node[i] == self.parent_node[id]:
                s.add(i)
        return s

def topological_path_dfs(mst, path, visited, prev, node, end):
    if not visited[node]:
        visited[node] = True
        path.append(node)

        if node == end:  # escape
            return True

        neighbours = mst[node][0]
        if prev != node:
            neighbours.remove(prev)
            neighbours.append(prev)

        for neighbour in neighbours:
            escape = topological_path_dfs(mst, path, visited, node, neighbour, end)
            if escape:
                return path
    else:
        path.pop()  # remove last element


def topological_dist_dfs(mst, path):
    dist = 0.0
    for i in range(len(path)-1):
        ix = mst[path[i]][0].index(path[i+1])
        dist += mst[path[i]][1][ix]
    return dist

def topological_distances(mst, hub_idx):
    n = len(hub_idx)
    dists = []
    for i in range(n):
        print(f"topological distance {i}/{n}")
        for j in range(i):
            path = []
            visited = [False] * len(mst)
            path = topological_path_dfs(mst, path, visited, hub_idx[i], hub_idx[i], hub_idx[j])
            dist = topological_dist_dfs(mst, path)
            dists.append(dist)

    # calculate adjacency matrix
    adj = np.zeros((n, n))
    tril = np.tril_indices(n, -1)  # without diagonals
    adj[tril] = dists
    return adj + adj.T