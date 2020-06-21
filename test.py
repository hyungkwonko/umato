import umato
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


if __name__ == "__main__":
    x = load_digits()  # (1797, 64 dim)

    # t-sne
    embedding = TSNE(verbose=True).fit_transform(x.data)

    # UMAP
    embedding = umap.UMAP(verbose=True).fit_transform(x.data)

    print(embedding)
