import argparse
import os
import numpy as np
from scipy.stats import loguniform
from src.evaluation.models.dataset import save_csv
import matplotlib.pyplot as plt
import time

# plot and save the image in figures/ directory
def plot_bench(data, label, dir, name):

    plt.scatter(data[:, 0], data[:, 1], s=2.0, c=label, cmap="Spectral", alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(11))
    plt.title("Embedded")
    plt.savefig(f"figures/{dir}/{name}.png")
    plt.close()

# import given algorithm class dynamically
# input: module name, class name, hyperparameters file name (optional)
parser = argparse.ArgumentParser(description="DR algorithm benchmark")
parser.add_argument('-m', "--module", type=str, help="a module name including target class", required=True)
parser.add_argument('-c', "--classname", type=str, help="a class name that activate DR algorithm", required=True)
parser.add_argument('-p', "--paramfile", type=str, help="a file containing hyperparameters", default=None)
parser.add_argument('-d', "--dataset", nargs='+', help="a dataset name", default=None)
parser.add_argument('-r', "--repeat", type=int, help="number of times to repeat", default=1)

args = parser.parse_args()

# alg_class: an executable model class
# Assume the class has fit_transform method
alg_module = __import__(args.module, globals(), locals(), [args.classname], 0)
alg_class = getattr(alg_module, args.classname)

# get hyperparameters from console input or param file
hp_dict = {}
# if paramfile is not given
if args.paramfile is None:
    print("Input parameters in [name]=<value> form. If you want to end, press Enter twice.")
    s = input()
    while s and s != "":
        if '=' not in s:
            print("Wrong input format. Please write in [name]=<value> form.")
            s = input()
            continue
        s_list = s.split('=')
        # try to convert value into int, float, or boolean if possible
        try:
            print(s_list[1], type(s_list[1]))
            s_list[1] = int(s_list[1])
        except:
            try:
                print(s_list[1], type(s_list[1]))
                s_list[1] = float(s_list[1])
            except:
                try:
                    print(s_list[1], type(s_list[1]))
                    s_list[1] = bool(s_list[1])
                except:
                    print(s_list[1], type(s_list[1]))
        hp_dict[s_list[0]] = s_list[1]
        s = input()
# if paramfile is given
else:
    param_file = open(args.paramfile, 'r')
    for line in param_file.readlines():
        if '=' not in line:
            continue
        line = line.strip()
        s_list = line.split('=')
        # try to convert value into int, float, or boolean if possible
        try:
            s_list[1] = int(s_list[1])
        except:
            try:
                s_list[1] = float(s_list[1])
            except:
                try:
                    s_list[1] = bool(s_list[1])
                except:
                    pass
        hp_dict[s_list[0]] = s_list[1]

# get the list of names of datasets that will be used to test the algorithm
# if --dataset is not given, then use all datasets
dataset_list = []
if args.dataset is None:
    rootdir = 'umato_exp/datasets/npy'
    for rootdir, dirs, files in os.walk(rootdir):
        dataset_list = dirs
        break
else:
    dataset_list = args.dataset

# load dataset with .npy file and run algorithm for specified number of times
for datadir in dataset_list:
    print(f'Run [{args.classname}] as ')
    x = np.load(f'umato_exp/datasets/npy/{datadir}/data.npy')
    label = np.load(f'umato_exp/datasets/npy/{datadir}/label.npy')
    
    elapsed_time = []
    for i in range(args.repeat):
        start = time.time()
        y = alg_class(**hp_dict).fit_transform(x)
        end = time.time()
        print(f"[{args.classname}, {datadir}] elapsed time: {end-start}")
        elapsed_time.append()
    
    print(f"[{args.classname}, {datadir}] average time of {args.repeat} trials: {end-start}")

    # save train results into csv and image
    path = os.path.join(os.getcwd(), "visualization", "public", "results", datadir)
    save_csv(path, alg_name=args.classname, data=y, label=label)
    plot_bench(y, label, datadir, args.classname)