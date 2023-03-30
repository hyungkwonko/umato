from sklearn.manifold import Isomap
import argparse
import os
import numpy as np
from dataset import get_data, save_csv
import time

parser = argparse.ArgumentParser(description="Isomap embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)
parser.add_argument("--hp", type=bool, help="whether to explore hyperparameter settings", default=False)
parser.add_argument("--n_samples", type=int, help="choose number of samples", default=1500)

args = parser.parse_args()


if __name__ == "__main__":

    alg_name = f"Isomap"
    # read data
    x, label = get_data(args.data, n_samples=args.n_samples)

    if args.hp:
        nn = np.arange(5, 55, 5)

        for i in range(len(nn)):
            # run ISOMAP

            start = time.time()
            y = Isomap(n_components=args.dim, n_jobs=-1, n_neighbors=nn[i]).fit_transform(x)
            end = time.time()

            print(f"{alg_name} elapsed time: {end-start}")

            # save as csv
            path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
            save_csv(path, alg_name=f"isomap_{nn[i]}", data=y, label=label)
    else:
        y = Isomap(n_components=args.dim, n_jobs=-1).fit_transform(x)
        path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
        save_csv(path, alg_name="isomap", data=y, label=label)
