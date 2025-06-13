========
Tutorial
========

This tutorial provides a guide to using ClumsyGrad.

Chapter 1: Understanding Tensors
================================

What is a Tensor?
-----------------

A tensor is a multi-dimensional array that can track gradients for automatic differentiation.

.. code-block:: python

   from clumsygrad.tensor import Tensor
   from clumsygrad.types import TensorType
   
   # Creating different types of tensors
   input_tensor = Tensor([1, 2, 3], tensor_type=TensorType.INPUT)
   parameter_tensor = Tensor([0.5, -0.3, 0.8], tensor_type=TensorType.PARAMETER)

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

   from clumsygrad import math

   data = Tensor([[1, 2, 3], [4, 5, 6]], tensor_type=TensorType.PARAMETER)
   
   total = math.sum(data, axis=0)  # Sum along columns: [5, 7, 9]
   mean = math.mean(data, axis=1)  # Mean along rows: [2.0, 5.0]

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

Next Steps
==========

* Explore the :doc:`api_reference` for detailed function documentation