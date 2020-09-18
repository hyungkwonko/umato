"""
This will compare embedding result quantitatively using functions in utils.py
"""

from .models.dataset import read_data
from .utils import GlobalMeasure, LocalMeasure
import pandas as pd
import argparse
import numpy as np

MEASURE_LOCAL_LIST = [
    "Spearman",
    "Trustworthiness",
    "Continuity",
    "MRRE",
]

ALGO_LIST = ["pca", "tsne", "umap", "topoae", "umato"]
DATA_LIST = ["spheres"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="quantitative comparison of the embedding result"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="choose dataset: spheres, mnist, fmnist, cifar10",
        default="spheres",
    )
    parser.add_argument(
        "--load", type=bool, help="load hubs", default=False
    )
    args = parser.parse_args()


    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        measures = MEASURE_LOCAL_LIST
        algorithms = []
        values = []
        for alg in ALGO_LIST:

            print(f"[INFO] Test on [{args.data}] dataset using [{alg}] w/ {k} neighbors")

            # read data & embedding result
            x, z, label = read_data(args.data, alg)

            if args.load:
                with open('./hubs.npy', 'rb') as f:
                    print("loading hubs")
                    hubs = np.load(f)
                    x = x[hubs]
                    z = z[hubs]
                    print(f"xlen: {len(x)}")

            lmeasure = LocalMeasure(x, z, k=k)

            algorithms.extend([alg] * len(MEASURE_LOCAL_LIST))
            

            spearman_val = lmeasure.spearmans_rho()
            trust_val = lmeasure.trustworthiness()
            conti_val = lmeasure.continuity()
            mrre_val = lmeasure.mrre()

            values.extend(
                [spearman_val, trust_val, conti_val, mrre_val,]
                # [trust_val, conti_val, mrre_val,]
            )

            result = pd.DataFrame(
                {"measure": measures, "algorithm": algorithms, "values": values}
            )
            result = result.pivot(index="measure", columns="algorithm", values="values")

        print(f"{result}\n")