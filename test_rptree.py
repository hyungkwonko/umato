import umato
import numpy as np


if __name__ == "__main__":

    dim = 100
    class_num = 10
    mean_diff = list(range(class_num))
    mean_diff = [e * 2 for e in mean_diff]
    cov_diff = list(range(1, class_num + 1))
    # cov_diff = [e * 5 + 1 for e in cov_diff]
    n = [1000] * 10
    np.random.shuffle(mean_diff)



    # cov_diff = [30, 30, 1, 10, 10]
    # mean_diff = [0, 5]
    # cov_diff = [1, 5]
    # n = [100, 9900]

    print(mean_diff)
    print(cov_diff)

    x = np.empty((0,dim))
    label = []

    for i in range(class_num):
        mean = np.array([np.random.random() + mean_diff[i] for _ in range(dim)])
        cov = np.eye(dim) * cov_diff[i]
        tmp = np.random.multivariate_normal(mean=mean, cov=cov, size=n[i])

        x = np.append(x, tmp, axis=0)
        label.append([i] * n[i])

    print(x[0])

    embedding = umato.UMATO(verbose=True).fit_transform(x)
    print(embedding)
