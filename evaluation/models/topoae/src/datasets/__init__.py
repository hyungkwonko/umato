"""Datasets."""
from .manifolds import Spheres
from .mnist import MNIST
from .kuzushiji_mnist import KuzushijiMNIST
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR
__all__ = ['Spheres', 'MNIST', 'KuzushijiMNIST', 'FashionMNIST', 'CIFAR']
