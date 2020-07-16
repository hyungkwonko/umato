import umato
from sklearn.datasets import load_digits
from evaluation.models.dataset import get_data, save_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def dsphere(n=100, d=2, r=1):

    data = np.random.randn(n, d+1)
    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])
    return data


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    #it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance= 2 * r / np.sqrt(d)

    shift_matrix = np.random.normal(0,variance,[n_spheres, d+1])

    spheres = [] 
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        sphere_shifted = sphere + shift_matrix[i,:]
        print(np.mean(np.sqrt(np.sum(sphere_shifted**2, 1))))
        spheres.append(sphere_shifted)
        n_datapoints += n_samples

    #Additional big surrounding sphere:
    n_samples_big = 10*n_samples #int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    print(np.mean(np.sqrt(np.sum(big**2, 1))))
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        plt.show()

    #Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints) 
    label_index=0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples
    
    return dataset, labels



if __name__ == "__main__":

    df, l = create_sphere_dataset(n_samples=500, d=30, r=5)#, plot=True)
    # exit()
    df = pd.DataFrame(df)
    x = df.to_numpy()

    # UMTO
    embedding = umato.UMATO(verbose=True, ll=l).fit_transform(x)