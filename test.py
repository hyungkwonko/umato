import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import os
import numpy as np
import time
from umato.umato_ import plot_tmptmp
import argparse

parser = argparse.ArgumentParser(description="args for umato")
parser.add_argument("--data", type=str, help="choose data: spheres, mnist, fmnist, kmnist", default="mnist")
parser.add_argument("--hub_num", type=int, help="choose number of hubs", default=300)
args = parser.parse_args()

if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data(args.data)  # spheres, mnist, fmnist, kmnist

    # UMTO
    t1 = time.time()
    embedding = umato.UMATO(verbose=True, ll=label, hub_num=args.hub_num, random_state=13).fit_transform(x)
    # embedding = umato.UMATO(verbose=True, ll=label).fit_transform(x)
    t2 = time.time()
    print(t2-t1)

    plot_tmptmp(embedding, label, "pic5_fin")
    save_csv('./', alg_name="umato", data=embedding, label=label)