#!/bin/bash
invalue="$1"
if [[ "$invalue" = "remove" ]]
then
    cd data/MNIST/raw
    bash download.sh $invalue
    cd ../../..

    cd data/FashionMNIST/raw
    bash download.sh $invalue
    cd ../../..

    cd data/cifar-10-batches-py
    bash download.sh $invalue
    cd ../..

else
    cd data/MNIST/raw
    bash download.sh
    cd ../../..

    cd data/FashionMNIST/raw
    bash download.sh
    cd ../../..

    cd data/cifar-10-batches-py
    bash download.sh
    cd ../..
fi