import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from umato.umato_ import plot_tmptmp

if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data("spheres")  # spheres, mnist, fmnist, cifar10

    # UMTO
    t1 = time.time()
    embedding = umato.UMATO(verbose=True, ll=label).fit_transform(x)
    t2 = time.time()
    print(t2-t1)

    plot_tmptmp(embedding, label, "pic5_fin")
    # save_csv('./', alg_name="umato", data=embedding, label=label)