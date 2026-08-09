import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clumsygrad import activation as act
from src.clumsygrad import math as cg_math
from src.clumsygrad.gradcheck import check_gradient
from src.clumsygrad.loss import mae_loss, mse_loss
from src.clumsygrad.tensor import Tensor, TensorType


def _param(data):
    return Tensor(np.array(data, dtype=np.float32), tensor_type=TensorType.PARAMETER)


def _const(data):
    return Tensor(np.array(data, dtype=np.float32))


CASES = {
    "transpose": (lambda t: t.T(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    "add_tensor_same_shape": (
        lambda t: t + _const([0.5, -0.3, 0.2]),
        [1.0, 2.0, 3.0],
    ),
    "add_broadcast": (
        lambda t: t + _const([0.1, 0.2, 0.3]),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ),
    "add_scalar": (lambda t: t + 3.5, [1.0, -2.0, 0.5]),
    "sub_tensor_same_shape": (
        lambda t: t - _const([0.1, 0.2, 0.3]),
        [1.0, 2.0, 3.0],
    ),
    "sub_broadcast": (
        lambda t: t - _const([0.1, 0.2, 0.3]),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ),
    "sub_scalar": (lambda t: t - 1.5, [1.0, 2.0, 3.0]),
    "mul_tensor_same_shape": (
        lambda t: t * _const([2.0, -1.0, 0.5]),
        [1.0, 2.0, 3.0],
    ),
    "mul_broadcast": (
        lambda t: t * _const([2.0, -1.0, 0.5]),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ),
    "mul_scalar": (lambda t: t * 2.5, [1.0, -2.0, 3.0]),
    "matmul": (
        lambda t: t @ _const([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        [[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]],
    ),
    "power": (lambda t: t**3, [1.0, 2.0, 1.5]),
    "negate": (lambda t: -t, [1.0, -2.0, 3.0]),
    "reshape": (lambda t: t.reshape((3, 2)), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    "sum_no_axis": (lambda t: cg_math.sum(t), [1.0, 2.0, 3.0]),
    "sum_with_axis": (
        lambda t: cg_math.sum(t, axis=0),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ),
    "mean_no_axis": (lambda t: cg_math.mean(t), [1.0, 2.0, 3.0, 4.0]),
    "abs": (lambda t: cg_math.abs(t), [1.5, -2.5, 3.5]),
    "sqrt": (lambda t: cg_math.sqrt(t), [1.0, 4.0, 9.0]),
    "exp": (lambda t: cg_math.exp(t), [0.0, 1.0, -1.0]),
    "log": (lambda t: cg_math.log(t), [1.0, 2.0, 3.0]),
    "sin": (lambda t: cg_math.sin(t), [0.3, 1.2, -0.7]),
    "cos": (lambda t: cg_math.cos(t), [0.3, 1.2, -0.7]),
    "tan": (lambda t: cg_math.tan(t), [0.3, 0.6, -0.5]),
    "tanh": (lambda t: act.tanh(t), [0.3, -0.6, 1.2]),
    "relu": (lambda t: act.relu(t), [-2.0, 3.0, -1.0, 4.0]),
    "sigmoid": (lambda t: act.sigmoid(t), [0.3, -0.6, 1.2]),
    "softmax": (lambda t: act.softmax(t), [1.0, 2.0, 3.0]),
    "mse_loss": (lambda t: mse_loss(t, _const([1.0, 2.0, 3.0])), [1.5, 1.8, 3.2]),
    "mae_loss": (lambda t: mae_loss(t, _const([1.0, 2.0, 3.0])), [1.5, 1.8, 3.9]),
}


@pytest.mark.parametrize("fn, data", CASES.values(), ids=CASES.keys())
def test_gradient_matches_finite_difference(fn, data):
    tensor = _param(data)
    passed, max_abs_diff = check_gradient(fn, tensor)
    assert passed, f"Gradient mismatch: max abs diff {max_abs_diff}"


def test_check_gradient_raises_for_non_grad_tensor():
    x = Tensor([1.0, 2.0, 3.0], tensor_type=TensorType.INPUT)
    with pytest.raises(ValueError, match="must require gradients"):
        check_gradient(lambda t: cg_math.sum(t), x)


def test_check_gradient_catches_broken_backward(monkeypatch):
    import src.clumsygrad.grad as grad_module

    def broken_exp_backward(tensor, grad):
        return (grad * 2 * tensor._data,)

    monkeypatch.setattr(grad_module, "exp_backward", broken_exp_backward)

    def fn(t):
        from src.clumsygrad.tensor import Tensor as T

        new_tensor = T._create_node(
            data=np.exp(t._data),
            grad_fn=broken_exp_backward,
            parents=(t,),
        )
        return new_tensor

    tensor = _param([0.5, 1.0, -0.5])
    passed, _ = check_gradient(fn, tensor)
    assert not passed
