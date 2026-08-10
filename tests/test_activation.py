import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clumsygrad.activation import relu, sigmoid, softmax, tanh
from src.clumsygrad.tensor import Tensor, TensorType


class TestTanh:
    def test_forward(self):
        x = Tensor(np.array([0.0]), tensor_type=TensorType.PARAMETER)
        y = tanh(x)
        np.testing.assert_array_almost_equal(y.data, np.array([0.0]))

    def test_backward(self):
        x = Tensor(np.array([0.0]), tensor_type=TensorType.PARAMETER)
        y = tanh(x)
        y.backward()
        # derivative of tanh at 0 is 1 - tanh(0)^2 = 1
        np.testing.assert_array_almost_equal(x.grad, np.array([1.0]))


class TestRelu:
    def test_forward(self):
        x = Tensor(np.array([-2.0, 0.0, 3.0]), tensor_type=TensorType.PARAMETER)
        y = relu(x)
        np.testing.assert_array_almost_equal(y.data, np.array([0.0, 0.0, 3.0]))

    def test_backward(self):
        x = Tensor(np.array([-1.0, 2.0]), tensor_type=TensorType.PARAMETER)
        y = relu(x)
        y.backward(np.array([1.0, 1.0]))
        np.testing.assert_array_almost_equal(x.grad, np.array([0.0, 1.0]))


class TestSigmoid:
    def test_forward(self):
        x = Tensor(np.array([0.0]), tensor_type=TensorType.PARAMETER)
        y = sigmoid(x)
        np.testing.assert_array_almost_equal(y.data, np.array([0.5]))

    def test_backward(self):
        x = Tensor(np.array([0.0]), tensor_type=TensorType.PARAMETER)
        y = sigmoid(x)
        y.backward()
        # derivative of sigmoid at 0 is sigmoid(0) * (1 - sigmoid(0)) = 0.25
        np.testing.assert_array_almost_equal(x.grad, np.array([0.25]))


class TestSoftmax:
    def test_forward_sums_to_one(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), tensor_type=TensorType.PARAMETER)
        y = softmax(x)
        np.testing.assert_almost_equal(np.sum(y.data), 1.0, decimal=5)
        assert np.all(y.data > 0)

    def test_forward_uniform_input(self):
        x = Tensor(np.array([5.0, 5.0, 5.0]), tensor_type=TensorType.PARAMETER)
        y = softmax(x)
        np.testing.assert_array_almost_equal(y.data, np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_backward_shape(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), tensor_type=TensorType.PARAMETER)
        y = softmax(x)
        y.backward(np.array([0.1, 0.2, 0.3]))
        assert x.grad is not None
        assert x.grad.shape == x.shape
