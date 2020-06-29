#!/bin/bash
remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing CIFAR-10 Dataset..."
  find . -type f -not -name '*.sh' -print0 | xargs -0 rm --
  echo "Removed!"
else
  echo "Downloading CIFAR-10 Dataset..."
  wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" -O cifar-10-python.tar.gz
  echo "Extracting CIFAR-10..."
  tar -xvf cifar-10-python.tar.gz
  mv cifar-10-batches-py/* ./
  rm -rf cifar-10-batches-py
  rm cifar-10-python.tar.gz
  echo "Download Finished!"
fi