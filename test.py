import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import os
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data("spheres")  # spheres, mnist, fmnist, cifar10

    # Synthetic data to check the # of connected components
    # import numpy as np
    # x = np.array([[1,1,1,1,1]]*50 + [[7,7,7,7,7]]*50 + [[-6,-6,-6,-6,-6]]*50)

    # UMTO
    t1 = time.time()
    embedding = umato.UMATO(verbose=True, ll=label).fit_transform(x)
    t2 = time.time()

    print(t2-t1)


    plt.scatter(embedding[:, 0], embedding[:, 1], s=1.0, c=label, cmap="Spectral", alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(10))
    plt.title("Embedded")
    plt.savefig(f"./tmp/pic4.png")
    plt.close()


    # save_csv('./', alg_name="umato", data=embedding, label=label)
    print(embedding)
