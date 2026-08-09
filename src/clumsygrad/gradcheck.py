"""
Numerical gradient-checking utilities.
"""

from collections.abc import Callable

import numpy as np

from .tensor import Tensor


def check_gradient(
    fn: Callable[[Tensor], Tensor],
    tensor: Tensor,
    eps: float = 1e-3,
    tol: float = 1e-2,
) -> tuple[bool, float]:
    r"""
    Numerically verify the analytical gradient of `fn` with respect to `tensor`.

    `fn` is called with `tensor` and must return a `Tensor` built from it. If
    the output has more than one element, the analytical gradient is computed
    with an incoming gradient of ones (equivalent to backpropagating from
    ``sum(fn(tensor))``), and each finite-difference probe uses the same
    element-sum so the two stay comparable:

    .. math::
        \frac{\partial \sum \text{fn}(x)}{\partial x_i} \approx
        \frac{\sum \text{fn}(x_i + \epsilon) -
        \sum \text{fn}(x_i - \epsilon)}{2\epsilon}

    Args:
        fn: A function taking `tensor` and returning a Tensor derived from it.
        tensor: The tensor (must have `requires_grad=True`) to check gradients for.
        eps: Perturbation size for the central-difference approximation.
        tol: Maximum allowed max-absolute-difference between the analytical
            and numerical gradients for the check to pass.

    Returns:
        A tuple `(passed, max_abs_diff)`.

    Raises:
        ValueError: If `tensor` does not require gradients.
    """
    if not tensor.requires_grad:
        raise ValueError(
            "tensor must require gradients (use TensorType.PARAMETER) to "
            "check its gradient"
        )

    def scalarize(output: Tensor) -> float:
        return float(np.sum(output.data))

    tensor.grad = None
    output = fn(tensor)
    output.backward(np.ones_like(output.data))

    if tensor.grad is None:
        raise ValueError("backward() did not populate a gradient for tensor")

    analytical_grad = tensor.grad.copy()

    numerical_grad = np.zeros_like(tensor.data, dtype=np.float64)
    for index in np.ndindex(tensor.data.shape):
        original_value = tensor.data[index]

        tensor.data[index] = original_value + eps
        plus = scalarize(fn(tensor))

        tensor.data[index] = original_value - eps
        minus = scalarize(fn(tensor))

        tensor.data[index] = original_value

        numerical_grad[index] = (plus - minus) / (2 * eps)

    tensor.grad = analytical_grad
    max_abs_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
    return max_abs_diff <= tol, max_abs_diff
