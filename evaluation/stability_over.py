from scipy.spatial import procrustes
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="args")
parser.add_argument("--data", type=str, help="choose dataset", required=True, default='spheres')

args = parser.parse_args()

if __name__ == "__main__":

    algos = ['tsne', 'umap', 'umato']
    # algos = ['umap', 'umato']

    for algo in algos:
        print(f"run algorithm: {algo}")
        path = os.path.join(os.getcwd(), "visualization", "public", "results", "init", args.data)

        d1= pd.read_csv(os.path.join(path, f'{algo}_{args.data}_class.csv'))
        d2= pd.read_csv(os.path.join(path, f'{algo}_{args.data}_random.csv'))
        d3= pd.read_csv(os.path.join(path, f'{algo}_{args.data}_spectral.csv'))
        d4= pd.read_csv(os.path.join(path, f'{algo}_{args.data}_pca.csv'))

        d1_no = d1.drop(columns=['label'])
        d1_no = np.array(d1_no)
        d2_no = d2.drop(columns=['label'])
        d2_no = np.array(d2_no)
        d3_no = d3.drop(columns=['label'])
        d3_no = np.array(d3_no)
        d4_no = d4.drop(columns=['label'])
        d4_no = np.array(d4_no)

        num = len(d1_no)
        ix = np.arange(num)

        ds = [d1_no, d2_no, d3_no, d4_no]
        disparities = []
        for i in enumerate(ds):
            for j in enumerate(ds):
                if i == j:
                    continue
                else:
                    _, _, disparity = procrustes(i[1], j[1])
                    disparities.append(disparity)
        print(f"mean value: {np.mean(disparities)}")
        print("====")