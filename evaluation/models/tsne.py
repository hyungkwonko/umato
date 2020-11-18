from sklearn.manifold import TSNE
import argparse
import os
import numpy as np
from scipy.stats import loguniform
from .dataset import get_data, save_csv
from umato.umato_ import plot_tmptmp

parser = argparse.ArgumentParser(description="t-SNE embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)
parser.add_argument("--hp", type=bool, help="whether to explore hyperparameter settings", default=False)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data)

    if args.hp:
        # learning_rate = np.sort(loguniform.rvs(10, 1000, size=1000))[99::100]
        learning_rate = np.array([15.24742297, 23.48066375, 37.34107189, 58.27652395, 87.24048423, 137.33961493, 211.00561713, 374.36120544, 576.90813121, 983.37544116])
        perplexity = np.arange(5, 55, 5)

        for i in range(len(learning_rate)):
            for j in range(len(perplexity)):

                # run TSNE
                y = TSNE(n_components=args.dim, perplexity=perplexity[j], learning_rate=learning_rate[i], init="pca", n_jobs=-1, random_state=0, verbose=2).fit_transform(x)

                # save as csv
                path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
                save_csv(path, alg_name=f"tsne_{perplexity[j]}_{learning_rate[i]}", data=y, label=label)
    else:
        y = TSNE(n_components=args.dim, random_state=0, verbose=2, init="pca", n_jobs=-1).fit_transform(x)
        path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
        save_csv(path, alg_name="tsne", data=y, label=label)
        plot_tmptmp(y, label, "tsne")
