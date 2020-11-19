import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import os
import numpy as np
from umato.umato_ import plot_tmptmp
from umato.utils import init_position
import argparse

parser = argparse.ArgumentParser(description="args for umato")
parser.add_argument("--data", type=str, help="choose data: spheres, mnist, fmnist, kmnist, flow, swissroll, scurve", default="spheres")
parser.add_argument("--hub_num", type=int, help="choose number of hubs", default=200)
parser.add_argument("--n_samples", type=int, help="choose number of samples", default=1500)
parser.add_argument("--init", type=str, help="choose initialization method", default="pca")
args = parser.parse_args()


if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data(args.data, n_samples=args.n_samples)  # spheres, mnist, fmnist, kmnist

    # x = x[np.arange(0, 10000, 50)]
    # label = label[np.arange(0, 10000, 50)]

    # UMTO
    for dt in ['mnist', 'fmnist', 'kmnist']:
        for mtd in ['spectral', 'pca', 'random', 'class']:
            init = init_position(x, label, dname=dt, init_type=mtd)
            y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, init=init).fit_transform(x)
            # y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, init="pca", global_learning_rate=0.0015, local_learning_rate=0.2).fit_transform(x)

            plot_tmptmp(y, label, "pic5_fin")
            # path = os.path.join(os.getcwd(), "visualization", "public", "results", dt)
            # save_csv(path, alg_name="umato", data=y, label=label)
            plot_tmptmp(y, label, f"pic_umato_{dt}_{mtd}")
            save_csv('./', alg_name=f"umato_{dt}_{mtd}", data=y, label=label)