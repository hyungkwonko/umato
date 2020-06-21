#!/bin/bash

invalue="$1"
if [[ "$invalue" = "" ]]
then
   for dname in spheres mnist fmnist cifar10;
      do
         echo "[RUN-TOPO-AE INFO] Training topological autoencoder model using $dname dataset..."
         python -m exp.train_model with settings/$dname.json
         echo "[RUN-TOPO-AE INFO] Finished training topological autoencoder model!"
         echo "[RUN-TOPO-AE INFO] Generating CSV file..."
         python -m exp.save_csv with settings/$dname.json
         echo "[RUN-TOPO-AE INFO] Finished generating CSV file!"
      done
else # if data name is given, run for a specific dataset
   echo "[RUN-TOPO-AE INFO] Training topological autoencoder model using $invalue dataset..."
   python -m exp.train_model with settings/$invalue.json
   echo "[RUN-TOPO-AE INFO] Finished training topological autoencoder model!"
   echo "[RUN-TOPO-AE INFO] Generating CSV file..."
   python -m exp.save_csv with settings/$invalue.json
   echo "[RUN-TOPO-AE INFO] Finished generating CSV file!"
fi

