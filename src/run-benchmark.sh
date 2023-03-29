#!/bin/bash
invalue="$1"
if [[ "$invalue" = "" ]]
then
   for dname in mnist spheres fmnist cifar10;
        do
            cd evaluation/models/topoae
            sh run-topoae.sh $dname
            cd ../../..

            echo "[RUN-PCA INFO] PCA embedding using $invalue dataset..."
            python3 evaluation/models/pca.py --data=$invalue
            echo "[RUN-PCA INFO] Finished PCA embedding!"

            echo "[RUN-T-SNE INFO] T-SNE embedding using $dname dataset..."
            python3 evaluation/models/tsne.py --data=$dname
            echo "[RUN-T-SNE INFO] Finished T-SNE embedding!"

            echo "[RUN-UMAP INFO] UMAP embedding using $dname dataset..."
            python3 evaluation/models/umap.py --data=$dname
            echo "[RUN-UMAP INFO] Finished UMAP embedding!"
        done
else # if data name is given, run for a specific dataset
    cd evaluation/models/topoae
    sh run-topoae.sh $invalue
    cd ../../..
    
    echo "[RUN-PCA INFO] PCA embedding using $invalue dataset..."
    python3 evaluation/models/pca.py --data=$invalue
    echo "[RUN-PCA INFO] Finished PCA embedding!"
``  
    echo "[RUN-ISOMAP INFO] ISOMAP embedding using $invalue dataset..."
    python3 evaluation/models/isomap.py --data=$invalue
    echo "[RUN-ISOMAP INFO] Finished ISOMAP embedding!"

    echo "[RUN-T-SNE INFO] T-SNE embedding using $invalue dataset..."
    python3 evaluation/models/tsne.py --data=$invalue
    echo "[RUN-T-SNE INFO] Finished T-SNE embedding!"

    echo "[RUN-UMAP INFO] UMAP embedding using $invalue dataset..."
    python3 evaluation/models/umap__.py --data=$invalue
    echo "[RUN-UMAP INFO] Finished UMAP embedding!"

fi