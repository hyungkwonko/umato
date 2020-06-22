from sklearn.manifold import TSNE
import argparse
import os
from .dataset import get_data
import pandas as pd

parser = argparse.ArgumentParser(description="t-SNE embedding")
parser.add_argument("--data", type=str, help="choose dataset", required=True)
parser.add_argument("--dim", type=str, help="choose embedding dimension", default=2)

args = parser.parse_args()


if __name__ == "__main__":

    x, label = get_data(args.data)

    # run TSNE
    y = TSNE(n_components=args.dim, random_state=0, verbose=1).fit_transform(x)

    # save as csv
    df = pd.DataFrame(y)
    df['label'] = label
    df.to_csv(os.path.join(os.getcwd(), 'results', args.data, 'tsne.csv'), index=False)
