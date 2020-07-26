"""
This will compare embedding result quantitatively using functions in utils.py
"""

from .models.dataset import read_data
from .utils import Measure
import pandas as pd
import argparse

MEASURE_GLOBAL_LIST = [
    "RMSE",
    "Kruskal",
    "Sammon",
    "DTM",
    "DTM_KL1",
    "DTM_KL01",
    "DTM_KL001",
]

MEASURE_LOCAL_LIST = [
    "Spearman",
    "Trustworthiness",
    "Continuity",
    "MRRE",
]

ALGO_LIST = ["pca", "tsne", "umap", "topoae", "umato"]  # (TODO) add umato
# DATA_LIST = ["spheres", "mnist", "fmnist", "cifar10"]
ALGO_LIST = ["umato"]  # (TODO) add umato
DATA_LIST = ["spheres"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="quantitative comparison of the embedding result")
    parser.add_argument("--algo", type=str, help="choose algorithm: pca, tsne, umap, topoae, umato", default="all")
    parser.add_argument("--data", type=str, help="choose dataset: spheres, mnist, fmnist, cifar10", default="spheres",)
    parser.add_argument("--measure", type=str, help="choose measures: all, global, local", default="all")
    args = parser.parse_args()

    measures = []
    algorithms = []
    values = []

    if args.algo 

    for n in ALGO_LIST:
        print(f"[INFO] Test on [{args.data}] dataset using [{n}]")
        # read data & embedding result
        x, z, label = read_data(args.data, n)
        # x = x[:100]
        # z = z[:100]

        mc = Measure(x, z, k=5)

        algorithms.extend([n] * len(MEASURE_LIST))
        measures.extend(MEASURE_LIST)

        rmse_val = mc.rmse()
        kruskal_val = mc.kruskal_stress_measure()
        sammon_val = mc.sammon_stress()
        spearman_val = mc.spearmans_rho()
        trust_val = mc.trustworthiness()
        conti_val = mc.continuity()
        mrre_val = mc.mrre()
        dtm_val = mc.dtm()
        dtmkl1_val = mc.dtm_kl(sigma=1.0)
        dtmkl01_val = mc.dtm_kl(sigma=0.1)
        dtmkl001_val = mc.dtm_kl(sigma=0.01)
        values.extend(
            [
                rmse_val,
                kruskal_val,
                sammon_val,
                spearman_val,
                trust_val,
                conti_val,
                mrre_val,
                dtm_val,
                dtmkl1_val,
                dtmkl01_val,
                dtmkl001_val,
            ]
        )

    print(f"[INFO] dataset [{args.data}] - Quantitative result")
    result = pd.DataFrame(
        {"measure": measures, "algorithm": algorithms, "values": values}
    )
    result = result.pivot(
        index="measure", columns="algorithm", values="values"
    ).fillna("NA")
    print(f"{result}\n")
