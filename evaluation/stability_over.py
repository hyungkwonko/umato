from scipy.spatial import procrustes
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    algos = ['tsne', 'umap', 'umato']
    data = 'mnist'

    for algo in algos:
        print(f"run algorithm: {algo}")
        path = os.path.join(os.getcwd(), "visualization", "public", "results", "init")

        d1= pd.read_csv(os.path.join(path, f'{algo}_{data}_class.csv'))
        d2= pd.read_csv(os.path.join(path, f'{algo}_{data}_random.csv'))
        d3= pd.read_csv(os.path.join(path, f'{algo}_{data}_pca.csv'))

        d1_no = d1.drop(columns=['label'])
        d1_no = np.array(d1_no)
        d2_no = d2.drop(columns=['label'])
        d2_no = np.array(d2_no)
        d3_no = d3.drop(columns=['label'])
        d3_no = np.array(d3_no)

        num = len(d1_no)
        ix = np.arange(num)

        ds = [d1_no, d2_no, d3_no]
        for i in ds:
            for j in ds:
                _, _, disparity = procrustes(i, j)
                print(disparity)
        print("====")