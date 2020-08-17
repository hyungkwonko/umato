from .models.dataset import read_data
from .utils import GlobalMeasure
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="quantitative comparison of the embedding result"
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="choose algorithm: pca, tsne, umap, topoae, umato",
        default="all",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="choose dataset: spheres, mnist, fmnist, cifar10",
        default="spheres",
    )
    args = parser.parse_args()

    # read data & embedding result
    x, z, label = read_data(args.data, args.algo)
    gmeasure = GlobalMeasure(x, z)
    dtmkl01_val = gmeasure.dtm_kl(sigma=0.1)
    print(f"DTM_KL01\t{dtmkl01_val}")