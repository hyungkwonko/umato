#!/bin/bash
invalue="$1"
if [[ "$invalue" = "" ]]
then
   for dname in spheres mnist fmnist cifar10;
        do
            cd models/topoae
            sh run-topoae.sh $dname
            cd ../..

            echo "[RUN-T-SNE INFO] T-SNE embedding using $dname dataset..."
            python -m models.tsne --data=$dname
            echo "[RUN-T-SNE INFO] Finished T-SNE embedding!"

            echo "[RUN-UMAP INFO] UMAP embedding using $dname dataset..."
            python -m models.umap --data=$dname
            echo "[RUN-UMAP INFO] Finished UMAP embedding!"
        done
else # if data name is given, run for a specific dataset
    cd models/topoae
    sh run-topoae.sh $invalue
    cd ../..

    echo "[RUN-T-SNE INFO] T-SNE embedding using $invalue dataset..."
    python -m models.tsne --data=$invalue
    echo "[RUN-T-SNE INFO] Finished T-SNE embedding!"

    echo "[RUN-UMAP INFO] UMAP embedding using $invalue dataset..."
    python -m models.umap --data=$invalue
    echo "[RUN-UMAP INFO] Finished UMAP embedding!"
fi

