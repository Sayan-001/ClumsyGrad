"""
This module provides basic mathematical operations.
"""

import numpy as np

from .tensor import Tensor


def sum(
    tensor: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """
    Compute the sum of the tensor along specified axis.

    Args:
        tensor: The input Tensor to compute the sum of.
        axis: Axis or axes along which the sum is computed. Default is None, which computes the sum of the flattened array.
        keepdims: If True, the reduced axes are left in the result as dimensions with size one.

    Returns:
        A new Tensor containing the sum of the input tensor.
    """

    from .grad import sum_backward

    new_tensor = Tensor._create_node(
        # numpy's overloads key off `keepdims` as a Literal; a plain `bool` can't
        # select one, so the call is untypeable without narrowing keepdims first.
        data=np.sum(tensor._data, axis=axis, keepdims=keepdims),  # type: ignore[call-overload]
        grad_fn=sum_backward,
        parents=(tensor,),
        extra={"axis": axis, "keepdims": keepdims, "input_shape": tensor._shape},
    )
    return new_tensor


def mean(
    tensor: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """
    Compute the mean of the tensor along specified axis.

    Args:
        tensor: The input Tensor to compute the mean of.
        axis: Axis or axes along which the means are computed. Default is None, which computes the mean of the flattened array.
        keepdims: If True, the reduced axes are left in the result as dimensions with size one.

    Returns:
        A new Tensor containing the mean of the input tensor.
    """

    from .grad import mean_backward

    new_tensor = Tensor._create_node(
        # See the comment on the `np.sum` call above re: keepdims typing.
        data=np.mean(tensor._data, axis=axis, keepdims=keepdims),  # type: ignore[call-overload]
        grad_fn=mean_backward,
        parents=(tensor,),
        extra={"axis": axis, "keepdims": keepdims, "input_shape": tensor._shape},
    )
    return new_tensor


def abs(tensor: Tensor) -> Tensor:
    """
    Compute the absolute value of the tensor.
    """

    from .grad import abs_backward

    new_tensor = Tensor._create_node(
        data=np.abs(tensor._data),
        grad_fn=abs_backward,
        parents=(tensor,),
    )
    return new_tensor


def sqrt(tensor: Tensor) -> Tensor:
    """
    Compute the square root of the tensor.
    """

    from .grad import sqrt_backward

    new_tensor = Tensor._create_node(
        data=np.sqrt(tensor._data),
        grad_fn=sqrt_backward,
        parents=(tensor,),
    )
    return new_tensor


def exp(tensor: Tensor) -> Tensor:
    """
    Compute the exponential of the tensor.
    """

    from .grad import exp_backward

    new_tensor = Tensor._create_node(
        data=np.exp(tensor._data),
        grad_fn=exp_backward,
        parents=(tensor,),
    )
    return new_tensor


def log(tensor: Tensor) -> Tensor:
    """
    Compute the natural logarithm of the tensor.
    """

    from .grad import log_backward

    new_tensor = Tensor._create_node(
        data=np.log(tensor._data),
        grad_fn=log_backward,
        parents=(tensor,),
    )
    return new_tensor


def sin(tensor: Tensor) -> Tensor:
    """
    Compute the sine of the tensor.
    """

    from .grad import sin_backward

    new_tensor = Tensor._create_node(
        data=np.sin(tensor._data),
        grad_fn=sin_backward,
        parents=(tensor,),
    )
    return new_tensor


def cos(tensor: Tensor) -> Tensor:
    """
    Compute the cosine of the tensor.
    """

    from .grad import cos_backward

    new_tensor = Tensor._create_node(
        data=np.cos(tensor._data),
        grad_fn=cos_backward,
        parents=(tensor,),
    )
    return new_tensor


def tan(tensor: Tensor) -> Tensor:
    """
    Compute the tangent of the tensor.
    """

    from .grad import tan_backward

    new_tensor = Tensor._create_node(
        data=np.tan(tensor._data),
        grad_fn=tan_backward,
        parents=(tensor,),
    )
    return new_tensor
