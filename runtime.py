# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP
from umato import UMATO
import argparse
import os
import numpy as np
from scipy.stats import loguniform
from .models.dataset import get_data, save_csv
import time

parser = argparse.ArgumentParser(description="t-SNE embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)

args = parser.parse_args()


if __name__ == "__main__":

    # read data
    x, label = get_data(args.data)

    # # run TSNE
    # times = []
    # for i in range(10):
    #     print(f"TSNE {i} th run")
    #     t1 = time.time()
    #     y = TSNE(n_jobs=40).fit_transform(x)
    #     t2 = time.time()
    #     times.append(t2-t1)
    
    # print("="*80)
    # print(f"tsne mean: {np.mean(times)}")
    # print(f"tsne std: {np.std(times)}")

    # # run UMAP
    # times = []
    # for i in range(10):
    #     print(f"UMAP {i} th run")
    #     t1 = time.time()
    #     y = UMAP().fit_transform(x)
    #     t2 = time.time()
    #     times.append(t2-t1)
    
    # print("="*80)
    # print(f"umap mean: {np.mean(times)}")
    # print(f"umap std: {np.std(times)}")

    # run UMATO
    times = []
    for i in range(10):
        print(f"UMATO {i} th run")
        t1 = time.time()
        y = UMATO(ll=label, hub_num=300, verbose=False).fit_transform(x)
        t2 = time.time()
        times.append(t2-t1)
    
    print("="*80)
    print(f"umato mean: {np.mean(times)}")
    print(f"umato std: {np.std(times)}")
