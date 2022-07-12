from sklearn.decomposition import PCA
import argparse
import os
from .dataset import get_data, save_csv
from umato.umato_ import plot_tmptmp

parser = argparse.ArgumentParser(description="PCA embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)
parser.add_argument("--n_samples", type=int, help="choose number of samples", default=1500)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data, n_samples=args.n_samples)

    # run PCA
    y = PCA(n_components=args.dim).fit_transform(x)

    # save as csv
    path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
    plot_tmptmp(y, label, "pca")
    save_csv(path, alg_name="pca", data=y, label=label)
