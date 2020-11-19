from umap import UMAP
import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
from .dataset import get_data, save_csv
from umato.utils import init_position
from umato.umato_ import plot_tmptmp

parser = argparse.ArgumentParser(description="UMAP embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)
parser.add_argument("--init", type=str, help="choose initialization method", default="pca")
parser.add_argument("--hp", type=bool, help="whether to explore hyperparameter settings", default=False)
parser.add_argument("--n_samples", type=int, help="choose number of samples", default=1500)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data, n_samples=args.n_samples)

    if args.hp:
        n_neighbor = np.arange(5, 55, 5)
        min_dist = np.arange(0, 1.1, 0.1)

        for i in range(len(n_neighbor)):
            for j in range(len(min_dist)):

                # run UMAP
                y = UMAP(n_components=args.dim, n_neighbors=n_neighbor[i], min_dist=min_dist[j], verbose=True).fit_transform(x)

                # save as csv
                path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
                save_csv(path, alg_name=f"umap_{n_neighbor[i]}_{min_dist[j]}", data=y, label=label)
    else:
        for dt in ['spheres', 'mnist', 'fmnist', 'kmnist']:
            for mtd in ['spectral', 'pca', 'random', 'class']:
                init = init_position(x, label, dname=dt, init_type=mtd)
                y = UMAP(n_components=args.dim, verbose=True, init=init).fit_transform(x)
                # y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, init="pca", global_learning_rate=0.0015, local_learning_rate=0.2).fit_transform(x)

                plot_tmptmp(y, label, f"pic_umap_{dt}_{mtd}")
                save_csv('./', alg_name=f"umap_{dt}_{mtd}", data=y, label=label)


        # init = init_position(x, label, init_type=args.init)
        # y = UMAP(n_components=args.dim, verbose=True, init="spectral").fit_transform(x)
        # path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
        # save_csv(path, alg_name="umap", data=y, label=label)
        # plot_tmptmp(y, label, f"pic_umap_{args.data}_{args.init}")
        # save_csv('./', alg_name=f"umap_{args.data}_{args.init}", data=y, label=label)
        # n_epochs = 200
        # init = PCA(n_components=args.dim).fit_transform(x)
        # y = UMAP(n_components=args.dim, verbose=True, n_epochs=n_epochs, init="spectral", gamma=).fit_transform(x)
