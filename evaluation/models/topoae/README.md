# Topological Autoencoder
The original source code is retrieved from [here](https://osf.io/abuce/?view_only=f16d65d3f73e4918ad07cdd08a1a0d4b), and modified by Hyung-Kwon Ko.

For further information on this work, you may refer to the original paper, [Topological Autoencoder (Moor et al., ICML 2020)](https://arxiv.org/abs/1906.00722).


## Training topological autoencoders
```
# dnames=[spheres, mnist, fmnist, cifar10]
python -m exp.train_model with settings/{dname}.json
```

## Generating .csv files
```
python -m exp.save_csv with settings/{dname}.json
```

## Run training & generating all at once
```
# run for a specific dataset (e.g., MNIST dataset)
sh run-topoae.sh mnist

# run for all datasets
sh run-topoae.sh
```