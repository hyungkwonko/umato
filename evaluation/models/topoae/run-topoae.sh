#!/bin/bash
for dname in spheres mnist fmnist cifar10;
do
   echo "[INFO] Training topological autoencoder model using $dname dataset..."
   python -m exp.train_model with settings/$dname.json
   echo "[INFO] Finished training topological autoencoder model!"
   echo "[INFO] Generating CSV file..."
   python -m exp.train_model with settings/$dname.json
   echo "[INFO] Finished generating CSV file!"
done