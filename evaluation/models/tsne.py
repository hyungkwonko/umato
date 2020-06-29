from sklearn.manifold import TSNE
import argparse
import os
from .dataset import get_data, save_csv

parser = argparse.ArgumentParser(description="t-SNE embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data)

    # run TSNE
    y = TSNE(n_components=args.dim, random_state=0, verbose=1).fit_transform(x)

    # save as csv
    path = os.path.join(os.getcwd(), "evaluation", "results", args.data)
    save_csv(path, alg_name="tsne", data=y, label=label)
