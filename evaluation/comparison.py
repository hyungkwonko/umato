"""
This will compare embedding result quantitatively using functions in utils.py
"""

from models.dataset import read_data
from utils import Measure
# from utils_copy import Measure
import pandas as pd

MEASURE_LIST = [
    "RMSE",
    "MRRE",
]
ALGO_LIST = ["tsne", "umap", "topoae"]  # (TODO) add umato
DATA_LIST = ["spheres", "mnist", "fmnist"]  # (TODO) add cifar-10

if __name__ == "__main__":

    for m in DATA_LIST:
        measures = []
        algorithms = []
        values = []

        for n in ALGO_LIST:
            # read data & embedding result
            x, z, label = read_data(m, n)
            x = x[:100]
            z = z[:100]

            mc = Measure(x, z, k=5)

            algorithms.extend([n] * len(MEASURE_LIST))
            measures.extend(MEASURE_LIST)

            zz  = mc.trustworthiness()
            print(zz)
            exit()

            rmse_val = mc.rmse()  # RMSE
            mrre_val = mc.mrre(k=5)  # MRRE (TODO) need checking...
            values.extend(
                [
                    rmse_val,
                    mrre_val,
                ]
            )

        print(f"[INFO] {m} - Quantitative result")
        result = pd.DataFrame(
            {"measure": measures, "algorithm": algorithms, "values": values}
        )
        result = result.pivot(
            index="measure", columns="algorithm", values="values"
        ).fillna("NA")
        print(f"{result}\n")
