import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clumsygrad.loss import mae_loss, mse_loss
from src.clumsygrad.tensor import Tensor, TensorType


class TestMseLoss:
    def test_forward(self):
        pred = Tensor(np.array([1.0, 2.0, 3.0]), tensor_type=TensorType.PARAMETER)
        target = Tensor(np.array([1.0, 2.0, 4.0]))
        loss = mse_loss(pred, target)
        # ((0)^2 + (0)^2 + (-1)^2) / 3
        np.testing.assert_almost_equal(loss.data, 1.0 / 3.0, decimal=5)

    def test_backward(self):
        pred = Tensor(np.array([1.0, 2.0]), tensor_type=TensorType.PARAMETER)
        target = Tensor(np.array([0.0, 0.0]))
        loss = mse_loss(pred, target)
        loss.backward()
        # d/dpred mean((pred-target)^2) = 2 * (pred - target) / n
        expected = 2 * (pred.data - target.data) / 2
        np.testing.assert_array_almost_equal(pred.grad, expected)

    def test_shape_mismatch_raises(self):
        pred = Tensor(np.array([1.0, 2.0]), tensor_type=TensorType.PARAMETER)
        target = Tensor(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            mse_loss(pred, target)


class TestMaeLoss:
    def test_forward(self):
        pred = Tensor(np.array([1.0, 2.0, 3.0]), tensor_type=TensorType.PARAMETER)
        target = Tensor(np.array([1.0, 0.0, 5.0]))
        loss = mae_loss(pred, target)
        # (0 + 2 + 2) / 3
        np.testing.assert_almost_equal(loss.data, 4.0 / 3.0, decimal=5)

    def test_backward(self):
        pred = Tensor(np.array([3.0, 1.0]), tensor_type=TensorType.PARAMETER)
        target = Tensor(np.array([1.0, 1.0]))
        loss = mae_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        # sign of (pred - target) scaled by 1/n
        np.testing.assert_array_almost_equal(pred.grad, np.array([0.5, 0.0]))

    def test_shape_mismatch_raises(self):
        pred = Tensor(np.array([1.0, 2.0]), tensor_type=TensorType.PARAMETER)
        target = Tensor(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            mae_loss(pred, target)
