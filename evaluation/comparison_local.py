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

ALGO_LIST = ["pca", "tsne", "umap", "topoae", "atsne", "umato"]
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
        "--algo",
        type=str,
        help="choose algorithm: pca, tsne, umap, topoae, atsne, umato",
        default="all",
    )
    args = parser.parse_args()

    if args.algo != "all":
        ALGO_LIST = [args.algo]

    for k in [5]:
    # for k in [5, 10, 15, 20, 30, 35, 40, 45, 50, 75, 100]:
    # for k in [3,4,5,6,7,8,9,10,11,12,13]:
        measures = []
        algorithms = []
        values = []
        for alg in ALGO_LIST:

            print(f"[INFO] Test on [{args.data}] dataset using [{alg}] w/ {k} neighbors")

            # read data & embedding result
            x, z, label = read_data(args.data, alg)

            lmeasure = LocalMeasure(x, z, k=k)

            algorithms.extend([alg] * len(MEASURE_LOCAL_LIST))
            measures.extend(MEASURE_LOCAL_LIST)

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
            result = result.pivot(index="measure", columns="algorithm", values="values").fillna(
                "NA"
            )
        print(f"{result}\n")