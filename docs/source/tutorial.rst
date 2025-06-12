========
Tutorial
========

This tutorial provides a comprehensive guide to using ClumsyGrad, from basic tensor operations to building complete neural networks.

Chapter 1: Understanding Tensors
================================

What is a Tensor?
-----------------

In ClumsyGrad, a tensor is a multi-dimensional array that can track gradients for automatic differentiation.

.. code-block:: python

   from clumsygrad.tensor import Tensor
   from clumsygrad.types import TensorType
   
   # Creating different types of tensors
   input_tensor = Tensor([1, 2, 3], tensor_type=TensorType.INPUT)
   parameter_tensor = Tensor([0.5, -0.3, 0.8], tensor_type=TensorType.PARAMETER)
   
   print(f"Input tensor: {input_tensor}")
   print(f"Parameter tensor: {parameter_tensor}")
   print(f"Requires grad: {parameter_tensor.requires_grad}")

Tensor Types
------------

ClumsyGrad supports three types of tensors:

**INPUT**: Data tensors that don't require gradients
   Used for input data, targets, and constants.

**PARAMETER**: Learnable parameters that require gradients
   Used for weights, biases, and other trainable parameters.

**INTERMEDIATE**: Temporary tensors created during operations
   Automatically created during computations.

Chapter 2: Basic Operations
===========================

Arithmetic Operations
---------------------

.. code-block:: python

   a = Tensor([2, 3, 4], tensor_type=TensorType.PARAMETER)
   b = Tensor([1, 2, 3], tensor_type=TensorType.PARAMETER)
   
   # Element-wise operations
   c = a + b      # Addition: [3, 5, 7]
   d = a * b      # Multiplication: [2, 6, 12]
   e = a ** 2     # Power: [4, 9, 16]
   
   # Scalar operations
   f = a + 5      # Broadcast addition: [7, 8, 9]

Matrix Operations
-----------------

.. code-block:: python

   # Matrix multiplication
   A = Tensor([[1, 2], [3, 4]], tensor_type=TensorType.PARAMETER)
   B = Tensor([[5, 6], [7, 8]], tensor_type=TensorType.PARAMETER)
   
   C = A @ B      # Matrix multiplication
   D = A.T()      # Transpose

Reduction Operations
--------------------

.. code-block:: python

   data = Tensor([[1, 2, 3], [4, 5, 6]], tensor_type=TensorType.PARAMETER)
   
   total = data.sum()           # Sum all elements
   row_sums = data.sum(axis=1)  # Sum along rows
   mean_val = data.mean()       # Mean of all elements

Chapter 3: Automatic Differentiation
====================================

Understanding Gradients
-----------------------

Gradients tell us how the output changes with respect to inputs.

.. code-block:: python

   # Simple function: f(x) = x^2
   x = Tensor([3.0], tensor_type=TensorType.PARAMETER)
   y = x ** 2
   
   # Compute gradient: df/dx = 2x
   y.backward()
   print(f"x = {x.data}, gradient = {x.grad}")  # Should be 6.0

Chain Rule in Action
--------------------

.. code-block:: python

   # Composite function: f(x) = (x^2 + 1)^3
   x = Tensor([2.0], tensor_type=TensorType.PARAMETER)
   
   # Forward pass
   u = x ** 2      # u = x^2
   v = u + 1       # v = u + 1 = x^2 + 1
   y = v ** 3      # y = v^3 = (x^2 + 1)^3
   
   # Backward pass
   y.backward()
   
   # df/dx = 3(x^2 + 1)^2 * 2x = 6x(x^2 + 1)^2
   expected_grad = 6 * 2 * (2**2 + 1)**2  # = 6 * 2 * 25 = 300
   print(f"Computed gradient: {x.grad}")
   print(f"Expected gradient: {expected_grad}")

Best Practices
==============

1. **Use appropriate tensor types**: INPUT for data, PARAMETER for learnable weights
2. **Clear gradients**: Always clear gradients before each training step
3. **Monitor memory**: Use memory management functions for long training runs
4. **Validate gradients**: Compare with numerical gradients during development
5. **Start simple**: Begin with basic operations before building complex models

Next Steps
==========

* Explore the :doc:`api_reference` for detailed function documentation