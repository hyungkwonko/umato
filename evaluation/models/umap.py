from umap import UMAP
import argparse
import os
from .dataset import get_data, save_csv

parser = argparse.ArgumentParser(description="UMAP embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data)

    # run UMAP
    y = UMAP(n_components=args.dim, verbose=True).fit_transform(x)

    # save as csv
    path = os.path.join(os.getcwd(), "results", args.data)
    save_csv(path, alg_name="umap", data=y, label=label)

