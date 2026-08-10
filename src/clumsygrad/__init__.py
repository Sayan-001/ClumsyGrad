"""
A simple automatic differentiation library built on top of NumPy.
It provides a `Tensor` class with support for building dynamic computation graphs.

For detailed documentation, refer: `https://clumsygrad.readthedocs.io/en/latest/`
"""

from importlib.metadata import version as _version

from . import activation, grad, gradcheck, loss, math, optimizer, random, tensor

__version__ = _version("clumsygrad")

__all__ = [
    "activation",
    "grad",
    "gradcheck",
    "loss",
    "math",
    "optimizer",
    "random",
    "tensor",
]
