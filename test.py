import umato
from sklearn.datasets import load_digits
from .models.dataset import get_data, save_csv
import os
import numpy as np
from umato.umato_ import plot_tmptmp
from umato.utils import init_position
import argparse

parser = argparse.ArgumentParser(description="args for umato")
parser.add_argument("--data", type=str, help="choose data: spheres, mnist, fmnist, kmnist, flow, swissroll, scurve, single-cell", default="allen")
parser.add_argument("--hub_num", type=int, help="choose number of hubs", default=400)
parser.add_argument("--n_samples", type=int, help="choose number of samples", default=1500)
parser.add_argument("--init", type=str, help="choose initialization method", default="pca")
args = parser.parse_args()


if __name__ == "__main__":

    x, label = get_data(args.data, n_samples=args.n_samples)
    y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num).fit_transform(x)
    plot_tmptmp(y, label, f"umato")
    save_csv('./', alg_name=f"umato", data=y, label=label)

    # x = x[np.arange(0, 10000, 50)]
    # label = label[np.arange(0, 10000, 50)]

    # for epoch in [200, 500, 1000, 2000, 5000]:
    #     x, label = get_data(args.data, n_samples=args.n_samples)  # spheres, mnist, fmnist, kmnist
    #     y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, global_n_epochs=epoch).fit_transform(x)
    #     plot_tmptmp(y, label, f"umato_{args.data}_{epoch}")
    #     save_csv('./', alg_name=f"umato_{args.data}_{epoch}", data=y, label=label)

    # # UMTO
    # for dt in ['fmnist', 'mnist', 'kmnist']:
    #     # x = load_digits()  # (1797, 64 dim)
    #     x, label = get_data(dt, n_samples=args.n_samples)  # spheres, mnist, fmnist, kmnist

    #     for mtd in ['spectral']:
    #         init = init_position(x, label, dname=dt, init_type=mtd)
    #         y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, init=init, local_learning_rate=0.1).fit_transform(x)
    #         # y = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, init="pca", global_learning_rate=0.0015, local_learning_rate=0.2).fit_transform(x)

    #         plot_tmptmp(y, label, "pic5_fin")
    #         # path = os.path.join(os.getcwd(), "visualization", "public", "results", dt)
    #         # save_csv(path, alg_name="umato", data=y, label=label)
    #         plot_tmptmp(y, label, f"pic_umato_{dt}_{mtd}")
    #         save_csv('./', alg_name=f"umato_{dt}_{mtd}", data=y, label=label)