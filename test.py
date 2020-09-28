import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import os
import numpy as np
from umato.umato_ import plot_tmptmp
import argparse

parser = argparse.ArgumentParser(description="args for umato")
parser.add_argument("--data", type=str, help="choose data: spheres, mnist, fmnist, kmnist, flow", default="mnist")
parser.add_argument("--hub_num", type=int, help="choose number of hubs", default=300)
args = parser.parse_args()

if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data(args.data)  # spheres, mnist, fmnist, kmnist

    # UMTO
    embedding = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, local_learning_rate=1.0).fit_transform(x)
    # embedding = umato.UMATO(verbose=True, ll=label).fit_transform(x)

    plot_tmptmp(embedding, label, "pic5_fin")
    save_csv('./', alg_name="umato", data=embedding, label=label)