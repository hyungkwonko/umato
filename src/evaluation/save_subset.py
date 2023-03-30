import fcsparser
import os
import numpy as np
import pandas as pd
from .models.dataset import get_data, save_csv
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP
from umato import UMATO

if __name__ == "__main__":

    seed = 3
    rs = np.random.RandomState(seed)
    percents = [1, 2, 5, 10, 20, 30, 50, 60, 80, 100]
    # percents = [100]
    x, label = get_data('flow')

    for percent in percents:
        p = int(x.shape[0] * (percent / 100))
        print(f"run for {percent} / 100")
        ix = rs.choice(x.shape[0], p, replace=False)
        x_subset = x[ix]
        label_subset = label[ix]

        # save raw csv for atsne
        save_csv('./', alg_name=f"raw_{percent}", data=x_subset, label=label_subset)

        # # load raw csv
        # x_subset = pd.read_csv(f"./raw_{percent}.csv")
        # label_subset = x_subset['label']
        # x_subset = x_subset.drop(columns=['label'])

        # run tsne
        y = TSNE(n_components=2, n_iter=1500, n_jobs=40, random_state=seed, verbose=2).fit_transform(x_subset)
        save_csv('./', alg_name=f"tsne_{percent}", data=y, label=label_subset)

        # # run umap
        # y = UMAP(n_components=2, verbose=True, random_state=seed).fit_transform(x_subset)
        # save_csv('./', alg_name=f"umap_{percent}", data=y, label=label_subset)

        # # run umato
        # y = UMATO(verbose=True, hub_num=300, random_state=seed, ll=label_subset).fit_transform(x_subset)
        # save_csv('./', alg_name=f"umato_{percent}", data=y, label=label_subset)