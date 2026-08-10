import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clumsygrad.math import abs, cos, exp, log, mean, sin, sqrt, sum, tan
from src.clumsygrad.tensor import Tensor, TensorType

# Backward-pass correctness for each of these functions is covered by the
# direct grad_module.*_backward unit tests in test_grad.py. These tests only
# cover the forward computation of the public API.


class TestSum:
    def test_forward_no_axis(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), tensor_type=TensorType.PARAMETER)
        y = sum(x)
        np.testing.assert_array_almost_equal(y.data, np.array(10.0))

    def test_forward_with_axis(self):
        x = Tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            tensor_type=TensorType.PARAMETER,
        )
        y = sum(x, axis=0)
        np.testing.assert_array_almost_equal(y.data, np.array([5.0, 7.0, 9.0]))


class TestMean:
    def test_forward_no_axis(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), tensor_type=TensorType.PARAMETER)
        y = mean(x)
        np.testing.assert_array_almost_equal(y.data, np.array(2.5))


class TestAbs:
    def test_forward(self):
        x = Tensor(np.array([-1.0, 0.0, 3.0]), tensor_type=TensorType.PARAMETER)
        y = abs(x)
        np.testing.assert_array_almost_equal(y.data, np.array([1.0, 0.0, 3.0]))


class TestSqrt:
    def test_forward(self):
        x = Tensor(np.array([1.0, 4.0, 9.0]), tensor_type=TensorType.PARAMETER)
        y = sqrt(x)
        np.testing.assert_array_almost_equal(y.data, np.array([1.0, 2.0, 3.0]))


class TestExp:
    def test_forward(self):
        x = Tensor(np.array([0.0, 1.0]), tensor_type=TensorType.PARAMETER)
        y = exp(x)
        np.testing.assert_array_almost_equal(y.data, np.array([1.0, np.e]), decimal=5)


class TestLog:
    def test_forward(self):
        x = Tensor(np.array([1.0, np.e]), tensor_type=TensorType.PARAMETER)
        y = log(x)
        np.testing.assert_array_almost_equal(y.data, np.array([0.0, 1.0]), decimal=5)


class TestTrig:
    def test_sin_forward(self):
        x = Tensor(np.array([0.0, np.pi / 2]), tensor_type=TensorType.PARAMETER)
        y = sin(x)
        np.testing.assert_array_almost_equal(y.data, np.array([0.0, 1.0]), decimal=5)

    def test_cos_forward(self):
        x = Tensor(np.array([0.0, np.pi / 2]), tensor_type=TensorType.PARAMETER)
        y = cos(x)
        np.testing.assert_array_almost_equal(y.data, np.array([1.0, 0.0]), decimal=5)

    def test_tan_forward(self):
        x = Tensor(np.array([0.0, np.pi / 4]), tensor_type=TensorType.PARAMETER)
        y = tan(x)
        np.testing.assert_array_almost_equal(y.data, np.array([0.0, 1.0]), decimal=5)
