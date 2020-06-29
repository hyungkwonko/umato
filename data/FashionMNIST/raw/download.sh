#!/bin/bash
remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Fashion MNIST Dataset..."
  rm *.gz
  echo "Removed!"
else
  echo "Downloading Fashion MNIST Dataset..."
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz" -O train-labels-idx1-ubyte.gz
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz" -O t10k-images-idx3-ubyte.gz
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz" -O t10k-labels-idx1-ubyte.gz
  echo "Download Finished!"
fi