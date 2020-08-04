remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Kuzushiji MNIST Dataset..."
  rm *.npz
  rm *.csv
  echo "Removed!"
else
  echo "Downloading Kuzushiji MNIST Dataset..."
  # wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
  # wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz" -O train-labels-idx1-ubyte.gz
  # wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz" -O t10k-images-idx3-ubyte.gz
  # wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz" -O t10k-labels-idx1-ubyte.gz

  wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist_classmap.csv" -O kmnist_classmap.csv
  wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz" -O kmnist-train-imgs.npz
  wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz" -O kmnist-train-labels.npz
  wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz" -O kmnist-test-imgs.npz
  wget "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz" -O kmnist-test-labels.npz
  echo "Download Finished!"
fi