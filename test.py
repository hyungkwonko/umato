import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv


if __name__ == "__main__":
    # x = load_digits()  # (1797, 64 dim)
    x, label = get_data("mnist")  # spheres, mnist, fmnist, cifar10

    # UMTO
    embedding = umato.UMATO(verbose=True).fit_transform(x.data)

    print(embedding)
