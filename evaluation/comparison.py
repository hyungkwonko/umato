"""
This will compare embedding result quantitatively using functions in util.py
"""

from models.dataset import read_data
import pandas as pd
import random  # remove after implementing all measures

MEASURE_LIST = ["RMSE", "MRRE", "TRUST", "Continuity", "KL-Div"]
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

            algorithms.extend([n] * 5)
            measures.extend(MEASURE_LIST)

            # calculate RMSE
            values.append(random.random())

            # calculate MRRE
            values.append(random.random())

            # calculate Trust
            values.append(random.random())

            # calculate Continuity
            values.append(random.random())

            # calculate KL-Div
            values.append(random.random())

        print(f"[INFO] {m} - Quantitative result")
        result = pd.DataFrame(
            {"measure": measures, "algorithm": algorithms, "values": values}
        )
        result = result.pivot(
            index="measure", columns="algorithm", values="values"
        ).fillna("NA")
        print(f"{result}\n")
