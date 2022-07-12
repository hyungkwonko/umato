'''
This convert data from txt to csv
'''

import argparse
import csv
import pandas as pd

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
    # lrs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # perps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # # change txt to csv
    # for lr in lrs:
    #     for perp in perps:
    #         with open(f'{args.data}_result_{lr}_{perp}.txt', 'r') as in_file:
    #             stripped = (line.strip() for line in in_file)
    #             lines = (line.split("\t") for line in stripped if line)
    #             with open(f'{args.data}_{lr}_{perp}.csv', 'w') as out_file:
    #                 writer = csv.writer(out_file)
    #                 writer.writerow(('0', '1'))
    #                 writer.writerows(lines)

    # # add label
    # for lr in lrs:
    #     for perp in perps:
    #         x = pd.read_csv(f'{args.data}_{lr}_{perp}.csv', float_precision='round_trip')
    #         df = pd.DataFrame(x)
    #         y = pd.read_csv(f'visualization/public/results/{args.data}/pca.csv', float_precision='round_trip')
    #         df2 = pd.DataFrame(y)
    #         df['label'] = df2['label']

    #         df.to_csv(f"atsne_{args.data}_{lr}_{perp}.csv", index=False)

    samples = [1,2,5,10,20,30,50,60,80,100]

    # change txt to csv
    for sample in samples:
        with open(f'atsne_{sample}.txt', 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split("\t") for line in stripped if line)
            with open(f'atsne_tmp{sample}.csv', 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(('0', '1'))
                writer.writerows(lines)

    for sample in samples:
        x = pd.read_csv(f'atsne_tmp{sample}.csv', float_precision='round_trip')
        df = pd.DataFrame(x)
        y = pd.read_csv(f'tsne_{sample}.csv', float_precision='round_trip')
        df2 = pd.DataFrame(y)
        df['label'] = df2['label']

        df.to_csv(f"atsne_{sample}.csv", index=False)
