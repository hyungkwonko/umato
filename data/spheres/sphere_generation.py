import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description="spheres data generation")
parser.add_argument("--fname", type=str, help="choose dataset", default="spheres_save")
parser.add_argument("--total_samples", type=int, help="choose total_samples", default=10000)
parser.add_argument("--n_spheres", type=int, help="choose number of inner spheres", default=10)
parser.add_argument("--d", type=int, help="choose dimension", default=101)
parser.add_argument("--r", type=int, help="choose inner sphere's radius", default=5)
parser.add_argument("--r_out", type=int, help="choose outer sphere's radius", default=25)
parser.add_argument("--var", type=float, help="choose variance btwn inner spheres", default=1.0)
parser.add_argument("--seed", type=int, help="choose seed", default=42)
parser.add_argument("--plot", type=bool, help="choose whether to save fig", default=False)

args = parser.parse_args()


def dsphere(n=100, d=2, r=1, noise=None):

    data = np.random.randn(n, d)

    # Normalization
    data = r * data / np.sqrt(np.sum(data ** 2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    return data


def create_sphere_dataset(total_samples=10000, d=100, n_spheres=10, r=5, r_out=25, var=1.0, seed=42, plot=False):
    np.random.seed(seed)

    variance = r / np.sqrt(d-1) * var

    shift_matrix = np.random.normal(0, variance, [n_spheres, d])

    spheres = []
    n_datapoints = 0
    n_samples = total_samples // (2 * n_spheres)

    for i in np.arange(n_spheres):
        sphere = dsphere(n=n_samples, d=d, r=r)
        sphere_shifted = sphere + shift_matrix[i, :]
        spheres.append(sphere_shifted)
        n_datapoints += n_samples

    # Big surrounding sphere:
    n_samples_big = total_samples - n_datapoints
    big = dsphere(n=n_samples_big, d=d, r=r_out)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres+1))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color], s=5.0)
        plt.savefig("sample.png")

    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index : label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    return dataset, labels


if __name__ == "__main__":
    d, l = create_sphere_dataset(total_samples=args.total_samples, n_spheres=args.n_spheres,
        d=args.d, r=args.r, r_out=args.r_out, var=args.var, seed=args.seed, plot=args.plot)
    df = pd.DataFrame(d)
    df["label"] = l

    # randomize data order
    # df = shuffle(df).reset_index(drop=True)
    df.to_csv(f"./{args.fname}.csv", index=False)
