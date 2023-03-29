'''
This convert data into txt (preparation for fvecs)
'''

import numpy as np
import argparse
from .models.dataset import get_data
# import pandas as pd

parser = argparse.ArgumentParser(
    description="data name"
)
parser.add_argument(
    "--data",
    type=str,
    help="choose dataset: spheres, mnist, fmnist, cifar10, flow",
    default="spheres",
)
args = parser.parse_args()


if __name__ == "__main__":
    # data, _ = get_data(args.data)
    # np.savetxt(f"./{args.data}.txt", data)
    # print(f"{args.data}: {data.shape} with type {type(data)} has been successfully saved!")

    import pandas as pd

    for i in [1,2,5,10,20,30,50,60,80,100]:
        data = pd.read_csv(f'./{args.data}_{i}.csv')
        data = data.drop(columns=['label'])
        np.savetxt(f"./{args.data}_{i}.txt", data)
        print(f"{args.data}_{i}: {data.shape} with type {type(data)} has been successfully saved!")