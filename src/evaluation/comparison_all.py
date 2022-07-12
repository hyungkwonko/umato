from .models.dataset import read_data, get_embed_data, get_data
from .utils import GlobalMeasure
import argparse
import glob
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="quantitative comparison of the embedding result"
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="choose algorithm: pca, tsne, umap, topoae, umato",
        default="all",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="choose dataset: spheres, mnist, fmnist, cifar10",
        default="spheres",
    )
    args = parser.parse_args()

    folder_path = f'./visualization/public/results/{args.data}'
    for filename in glob.glob(os.path.join(folder_path, f'{args.algo}*.csv')):
        with open(filename, 'r') as f:


            filenamelist = filename.split("/")
            filetarget = filenamelist[-1]
            c = filetarget.split(".")
            cc = c[0] + '.' +  c[1]
            if args.algo == "isomap":
                cc = c[0]

            # read data & embedding result
            z = get_embed_data(args.data, cc)
            x, label = get_data(args.data)

            gmeasure = GlobalMeasure(x, z)
            dtmkl01_val = gmeasure.dtm_kl(sigma=0.1)
            print(f"{cc}\tDTM_KL01\t{dtmkl01_val}")