'''
This convert data into txt (preparation for fvecs)
'''

import numpy as np
import argparse
from evaluation.models.dataset import get_data

parser = argparse.ArgumentParser(
    description="data name"
)
parser.add_argument(
    "--data",
    type=str,
    help="choose dataset: spheres, mnist, fmnist, cifar10",
    default="spheres",
)
args = parser.parse_args()


if __name__ == "__main__":
    data, _ = get_data(args.data)
    np.savetxt(f"./{args.data}.txt", data)
    print(f"{args.data}: {data.shape} with type {type(data)} has been successfully saved!")