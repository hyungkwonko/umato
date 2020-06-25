import umato
from sklearn.datasets import load_digits


if __name__ == "__main__":
    x = load_digits()  # (1797, 64 dim)

    # UMTO
    embedding = umato.UMAP(verbose=True).fit_transform(x.data)

    print(embedding)
