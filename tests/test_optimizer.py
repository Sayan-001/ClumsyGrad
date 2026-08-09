import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clumsygrad.optimizer import SGD, Adam
from src.clumsygrad.tensor import Tensor, TensorType


class TestOptimizerBase:
    def test_filters_non_parameter_tensors(self):
        param = Tensor(np.array([1.0]), tensor_type=TensorType.PARAMETER)
        non_param = Tensor(np.array([1.0]), tensor_type=TensorType.INPUT)
        opt = SGD([param, non_param], lr=0.1)
        assert opt.parameters == [param]

    def test_zero_grad_clears_gradients(self):
        param = Tensor(np.array([1.0, 2.0]), tensor_type=TensorType.PARAMETER)
        param.grad = np.array([0.5, 0.5])
        opt = SGD([param], lr=0.1)
        opt.zero_grad()
        assert param.grad is None


class TestSGD:
    def test_step_updates_parameter(self):
        param = Tensor(np.array([1.0, 2.0]), tensor_type=TensorType.PARAMETER)
        param.grad = np.array([1.0, 1.0])
        opt = SGD([param], lr=0.1)
        opt.step()
        np.testing.assert_array_almost_equal(param.data, np.array([0.9, 1.9]))

    def test_step_skips_params_without_grad(self):
        param = Tensor(np.array([1.0]), tensor_type=TensorType.PARAMETER)
        opt = SGD([param], lr=0.1)
        opt.step()
        np.testing.assert_array_almost_equal(param.data, np.array([1.0]))

    def test_reduces_toy_loss(self):
        # minimize (x - 3)^2 via manual gradient descent
        x = Tensor(np.array([0.0]), tensor_type=TensorType.PARAMETER)
        opt = SGD([x], lr=0.1)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            y = x - 3.0
            loss = y * y
            loss.backward()
            opt.step()
            losses.append(float(loss.data.item()))

        assert losses[-1] < losses[0]
        np.testing.assert_allclose(x.data, np.array([3.0]), atol=1e-2)


class TestAdam:
    def test_step_updates_parameter(self):
        param = Tensor(np.array([1.0]), tensor_type=TensorType.PARAMETER)
        param.grad = np.array([1.0])
        opt = Adam([param], lr=0.1)
        opt.step()
        # first step: m_hat = grad, v_hat = grad^2, so update = lr * 1 / (1 + eps) ~= lr
        assert param.data[0] < 1.0

    def test_step_skips_params_without_grad(self):
        param = Tensor(np.array([1.0]), tensor_type=TensorType.PARAMETER)
        opt = Adam([param])
        opt.step()
        np.testing.assert_array_almost_equal(param.data, np.array([1.0]))

    def test_reduces_toy_loss(self):
        x = Tensor(np.array([0.0]), tensor_type=TensorType.PARAMETER)
        opt = Adam([x], lr=0.1)

        losses = []
        for _ in range(100):
            opt.zero_grad()
            y = x - 3.0
            loss = y * y
            loss.backward()
            opt.step()
            losses.append(float(loss.data.item()))

        assert losses[-1] < losses[0]
        np.testing.assert_allclose(x.data, np.array([3.0]), atol=1e-1)
