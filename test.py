import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import os
import numpy as np
from umato.umato_ import plot_tmptmp
import argparse

parser = argparse.ArgumentParser(description="args for umato")
parser.add_argument("--data", type=str, help="choose data: spheres, mnist, fmnist, kmnist, flow, swissroll, scurve", default="spheres")
parser.add_argument("--hub_num", type=int, help="choose number of hubs", default=200)
parser.add_argument("--n_samples", type=int, help="choose number of samples", default=1500)
args = parser.parse_args()

if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data(args.data, n_samples=args.n_samples)  # spheres, mnist, fmnist, kmnist

    # x = x[np.arange(0, 10000, 50)]
    # label = label[np.arange(0, 10000, 50)]

    # UMTO
    embedding = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, random_state=2).fit_transform(x)
    # embedding = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, init="pca", global_learning_rate=0.0015, local_learning_rate=0.2).fit_transform(x)

    plot_tmptmp(embedding, label, "pic5_fin")
    path = os.path.join(os.getcwd(), "visualization", "public", "results", args.data)
    save_csv(path, alg_name="umato", data=embedding, label=label)
    # save_csv('./', alg_name=f"umato_{args.hub_num}", data=embedding, label=label)