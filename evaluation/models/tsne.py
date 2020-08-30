# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import argparse
import os
import numpy as np
from scipy.stats import loguniform
from .dataset import get_data, save_csv

parser = argparse.ArgumentParser(description="t-SNE embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)

args = parser.parse_args()


if __name__ == "__main__":

    # learning_rate = np.sort(loguniform.rvs(10, 1000, size=1000))[99::100]
    learning_rate = np.array([15.24742297, 23.48066375, 37.34107189, 58.27652395, 87.24048423, 137.33961493, 211.00561713, 374.36120544, 576.90813121, 983.37544116])
    perplexity = np.arange(5, 55, 5)

    for i in range(len(learning_rate)):
        for j in range(len(perplexity)):

            # read data
            x, label = get_data(args.data)

            # run TSNE
            y = TSNE(n_components=args.dim, perplexity=perplexity[j], learning_rate=learning_rate[i], n_iter=1500, n_jobs=40, random_state=0, verbose=2).fit_transform(x)

            # save as csv
            path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
            save_csv(path, alg_name=f"tsne_{perplexity[j]}_{learning_rate[i]}", data=y, label=label)
