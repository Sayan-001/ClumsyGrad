.. quickstart:

Quick Start Guide
=================

Installation
------------

To install ClumsyGrad, use pip:

.. code-block:: shell

   pip install clumsygrad

Basic Usage
-----------

Creating Tensors
~~~~~~~~~~~~~~~~

.. code-block:: python

   from clumsygrad.tensor import Tensor
   from clumsygrad.types import TensorType
   from clumsygrad import random
   
   # Create a tensor from a list (INPUT type by default)
   x = Tensor([[1.0, 2.0]])
   print(f"x: {x}")
   print(f"x requires_grad: {x.requires_grad}")  # False for INPUT type
   
   # Create a parameter tensor (requires_grad=True by default)
   w = Tensor([[0.5, 1.5]], tensor_type=TensorType.PARAMETER)
   print(f"w requires_grad: {w.requires_grad}")  # True for PARAMETER type
   
   # Random tensors
   z1 = random.rand((2, 3))  # Uniform random [0, 1)


Basic Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create parameter tensors for operations
   a = Tensor([[1.0, 2.0]])
   b = Tensor([[3.0, 4.0]])
   
   # Arithmetic operations
   add_result = a + b        # Element-wise addition
   sub_result = a - b        # Element-wise subtraction
   mul_result = a * b        # Element-wise multiplication
   
   # Scalar operations
   scalar_add = a + 5.0      # Add scalar to all elements
   scalar_mul = a * 2.0      # Multiply all elements by scalar
   
   # Matrix operations
   x = Tensor([[1.0, 2.0]])
   y = Tensor([[3.0], [4.0]])
   matmul_result = x @ y     # Matrix multiplication
   
   # Transpose
   transposed = x.T()

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple example: y = x^2 + 3x + 1
   x = Tensor([[2.0]], tensor_type=TensorType.PARAMETER)
   
   # Forward pass
   y = x ** 2 + 3 * x + 1
   print(f"y = {y.data}")  # Should be [15.0]
   
   # Backward pass
   y.backward()
   print(f"dy/dx = {x.grad}")  # Should be [7.0] (derivative: 2x + 3 = 4 + 3 = 7)

Best Practices
~~~~~~~~~~~~~~

1. **Use appropriate tensor types**: INPUT for data, PARAMETER for trainable weights
2. **Reset gradients**: Always reset gradients before backward pass in training loops
3. **Scalar outputs for backward()**: Call backward() only on scalar tensors (typically loss values)

For more usage and API documentation, see the :doc:`api_reference` section.