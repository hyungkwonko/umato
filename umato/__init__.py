from .umato_ import UMATO

# Workaround: https://github.com/numba/numba/issues/3341
import numba

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("umato")
except PackageNotFoundError:
    __version__ = "0.4-dev"