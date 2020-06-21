remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing MNIST Dataset..."
  rm *.gz
  echo "Removed!"
else
  echo "Downloading MNIST Dataset..."
  wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
  wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" -O train-labels-idx1-ubyte.gz
  wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" -O t10k-images-idx3-ubyte.gz
  wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" -O t10k-labels-idx1-ubyte.gz
  echo "Download Finished!"
fi