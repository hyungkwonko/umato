import umap
from sklearn.datasets import load_digits



if __name__ == "__main__":
    digits = load_digits()

    embedding = umap.UMAP(verbose=True).fit_transform(digits.data)
    print(embedding)