# Uniform Manifold Approximation with Two-phase Optimization

## What is UMATO?
work in progress...

### System Requirements
- Python 3.6 or greater
- scikit-learn
- numpy
- scipy
- numba
- pandas (to read csv data)

## Running benchmarks
You can try the following code to see the result:
```python
# install requirements
pipenv install

cd evaluation

# run all datasets
bash run.sh

# run specific dataset (e.g., MNIST dataset)
bash run.sh mnist
```

For detailed information, please refer to [here](https://github.com/hyungkwonko/umato/tree/master/evaluation).