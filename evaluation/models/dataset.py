import os
import pickle
import numpy as np
import pandas as pd
import gzip

# Load Kuzushiji Japanese Handwritten dataset
def load_kmnist(path, dtype="KMNIST", kind='train'):
    images_path = os.path.join(path, f'{dtype}-{kind}-imgs.npz')
    labels_path = os.path.join(path, f'{dtype}-{kind}-labels.npz')
    images = np.load(images_path)
    images = images.f.arr_0
    images = images.reshape(images.shape[0], -1)
    labels = np.load(labels_path)
    labels = labels.f.arr_0
    labels = labels.reshape(-1)
    return images, labels

# FASHION MNIST (60000+10000, 784), 26MB
def load_mnist(path, kind="train"):  # train, t10k

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


# CIFAR 10 (50000+10000, 3072), 163MB
def load_pickle(f):
    return pickle.load(f, encoding="latin1")


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3072)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(cifar10_dir):
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def get_data(dname):
    if dname == "spheres":
        path = os.path.join(os.getcwd(), "data", "spheres")
        df = pd.read_csv(os.path.join(path, 'spheres.csv')) # load data
        x = df.drop(columns=['label']).to_numpy()
        label = df['label'].to_numpy()
        return x, label
    elif dname == "mnist":
        path = os.path.join(os.getcwd(), "data", "MNIST", "raw")
        return load_mnist(path=path, kind="train")  # kind="t10k"
    elif dname == "fmnist":
        path = os.path.join(os.getcwd(), "data", "FashionMNIST", "raw")
        return load_mnist(path=path, kind="train")  # kind="t10k"
    elif dname == "kmnist":
        path = os.path.join(os.getcwd(), "data", "KMNIST", "raw")
        return load_kmnist(path=path, kind="train")  # kind="t10k"
    elif dname == "cifar10":
        path = os.path.join(os.getcwd(), "data", "cifar-10-batches-py")
        x, label, _, _ = get_CIFAR10_data(path)
        return x, label
    else:
        pass

def get_embed_data(dname, algo):
    if dname == "spheres":
        path = os.path.join(os.getcwd(), "evaluation", "results", "spheres")
        df = pd.read_csv(os.path.join(path, f'{algo}.csv')) # load data
        z = df.drop(columns=['label']).to_numpy()
        return z
    elif dname == "mnist":
        path = os.path.join(os.getcwd(), "evaluation", "results", "mnist")
        df = pd.read_csv(os.path.join(path, f'{algo}.csv')) # load data
        z = df.drop(columns=['label']).to_numpy()
        return z
    elif dname == "fmnist":
        path = os.path.join(os.getcwd(), "evaluation", "results", "fmnist")
        df = pd.read_csv(os.path.join(path, f'{algo}.csv')) # load data
        z = df.drop(columns=['label']).to_numpy()
        return z
    elif dname == "cifar10":
        path = os.path.join(os.getcwd(), "evaluation", "results", "cifar10")
        df = pd.read_csv(os.path.join(path, f'{algo}.csv')) # load data
        z = df.drop(columns=['label']).to_numpy()
        return z
    else:
        pass

def read_data(dname, algo):
    z = get_embed_data(dname, algo)
    x, label = get_data(dname)
    return x, z, label
    

def save_csv(path, alg_name, data, label):
    df = pd.DataFrame(data)
    df['label'] = label

    if not os.path.isdir(path):
        os.makedirs(path) # this makes directories including all paths

    df.to_csv(os.path.join(path, f'{alg_name}.csv'), index=False)
