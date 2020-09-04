'''
This convert data from txt to csv
'''

import argparse
import csv

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
    with open(f'{args.data}.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split("\t") for line in stripped if line)
        with open(f'{args.data}.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)