from sklearn.decomposition import PCA
import argparse
import os
from .dataset import get_data, save_csv

parser = argparse.ArgumentParser(description="PCA embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data)

    # run PCA
    y = PCA(n_components=args.dim).fit_transform(x)

    # save as csv
    path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
    save_csv(path, alg_name="pca", data=y, label=label)
