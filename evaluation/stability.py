from scipy.spatial import procrustes
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    datasets = ['tsne', 'umap', 'atsne', 'umato']
    datasets = ['atsne']
    samples = [1, 2, 5, 10, 20, 30, 50, 60, 80, 100]

    for dataset in datasets:
        print(f"run algorithm: {dataset}")
        for sample in samples:
            path = os.path.join(os.getcwd(), "visualization", "public", "results", "stability")

            d1= pd.read_csv(os.path.join(path, f'{dataset}_{sample}.csv'))
            d2= pd.read_csv(os.path.join(path, f'{dataset}_all.csv'))

            d1_label = d1['label']
            d1_no = d1.drop(columns=['label'])
            d1_no = np.array(d1_no)

            d2 = d2.sort_values(['label'], ascending=True)
            d2_no = d2.drop(columns=['label'])
            d2_no = np.array(d2_no)

            num = len(d1_no)
            ix = np.arange(num)

            _, _, disparity = procrustes(d1_no, d2_no[d1_label])

            # print(f"Disparity {sample}/100 % = {disparity}")
            print(disparity)