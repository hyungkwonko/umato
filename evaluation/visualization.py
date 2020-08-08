'''
This will include functions to visualize embedding result 
'''

import numpy as np
from evaluation.models.dataset import get_data, save_csv

def plot_tmptmp(data, label, name):
    import matplotlib.pyplot as plt

    plt.scatter(data[:, 0], data[:, 1], s=2.0, c=label, cmap="Spectral", alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(11))
    plt.title("Embedded")
    plt.savefig(f"./tmp/{name}.png")
    plt.close()


if __name__ == "__main__":
    x, label = get_data('kmnist')  # spheres, mnist, fmnist, cifar10
    plot_tmptmp(x, label, 'zzz')