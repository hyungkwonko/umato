from scipy.spatial import procrustes
import numpy as np
import pandas as pd
import os
import json


if __name__ == "__main__":

    datasets = ['tsne', 'umap', 'atsne', 'umato']
    datas = []

    for i in range(20):
        data = {}
        data[f'percentage'] = int(((i + 1) / 20) * 100)

        for dataset in datasets:
            path = os.path.join(os.getcwd(), "visualization", "public", "results", "stability")

            d1= pd.read_csv(os.path.join(path, f'{dataset}1.csv'))
            d2= pd.read_csv(os.path.join(path, f'{dataset}2.csv'))

            d1_no = d1.drop(columns=['label'])
            d1_no = np.array(d1_no)

            d2_no = d2.drop(columns=['label'])
            d2_no = np.array(d2_no)

            num = len(d1_no)
            ix = np.arange(num)

            result = []
            for j in range(10):
                sample_num = int(num * ((i + 1) / 20))
                ix_s = np.random.choice(ix, sample_num, replace=False)
                _, _, disparity = procrustes(d1_no[ix_s], d2_no[ix_s])
                result.append(disparity)
            data[f'{dataset}'] = np.mean(result)
            print(f"For {dataset}, {sample_num} / {num}: Procrustes Distance = {np.mean(result)} +- {np.std(result)}")

        datas.append(data)

    with open('./stability.json', 'w') as outfile:
        json.dump(datas, outfile)