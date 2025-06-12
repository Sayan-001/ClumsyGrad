"""
This module provides functions to compute various loss functions.
"""

from __future__ import annotations

from .tensor import Tensor
from .grad import mse_backward, mae_backward

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) loss between the predicted and target tensors.
    
    Args:
        pred (Tensor): The predicted tensor.
        target (Tensor): The target tensor.
        
    Returns:
        Tensor: The MSE loss tensor.
    """
    
    if pred._shape != target._shape:
        raise ValueError("Predicted and target tensors must have the same shape for MSE loss.")
    
    diff = pred - target
    mse = (diff * diff).mean()
    
    return Tensor._create_node(mse._data, 
                               parents=(pred, target), 
                               grad_fn=mse_backward, 
                               requires_grad=pred._requires_grad or target._requires_grad)
    
def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Computes the Mean Absolute Error (MAE) loss between the predicted and target tensors.
    
    Args:
        pred (Tensor): The predicted tensor.
        target (Tensor): The target tensor.
        
    Returns:
        Tensor: The MAE loss tensor.
    """
    
    if pred._shape != target._shape:
        raise ValueError("Predicted and target tensors must have the same shape for MAE loss.")
    
    diff = pred - target
    mae = diff.abs().mean()
    
    return Tensor._create_node(mae._data, 
                               parents=(pred, target), 
                               grad_fn=mae_backward,
                               requires_grad=pred._requires_grad or target._requires_grad)