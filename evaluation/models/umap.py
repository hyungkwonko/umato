from umap import UMAP
import argparse
import os
import numpy as np
from .dataset import get_data, save_csv

parser = argparse.ArgumentParser(description="UMAP embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)

args = parser.parse_args()


if __name__ == "__main__":

    n_neighbor = np.arange(5, 55, 5)
    min_dist = np.arange(0, 1.1, 0.1)

    for i in range(len(n_neighbor)):
        for j in range(len(min_dist)):

            # read data
            x, label = get_data(args.data)

            # run UMAP
            y = UMAP(n_components=args.dim, n_neighbors=n_neighbor[i], min_dist=min_dist[j], verbose=True).fit_transform(x)

            # save as csv
            path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
            save_csv(path, alg_name=f"umap_{n_neighbor[i]}_{min_dist[j]}", data=y, label=label)

