.. quickstart:

Quick Start Guide
=================

Installation
------------

Since ClumsyGrad is in development, install from source:

.. code-block:: bash

   git clone https://github.com/Sayan-001/ClumsyGrad.git
   cd ClumsyGrad
   pip install -e .

Basic Usage
-----------

Creating Tensors
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
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
   
   # Create from numpy array
   y = Tensor(np.array([[3.0, 4.0]]), tensor_type=TensorType.PARAMETER)
   
   # Random tensors
   z1 = random.rand((2, 3))  # Uniform random [0, 1)
   z2 = random.randn((2, 3))  # Normal distribution

Tensor Types
~~~~~~~~~~~~

ClumsyGrad supports three tensor types:

.. code-block:: python

   from clumsygrad.types import TensorType
   
   # INPUT: For data (requires_grad=False by default)
   data = Tensor([[1, 2, 3]], tensor_type=TensorType.INPUT)
   
   # PARAMETER: For trainable parameters (requires_grad=True by default)
   weights = Tensor([[0.1, 0.2]], tensor_type=TensorType.PARAMETER)
   
   # INTERMEDIATE: Created automatically during operations
   # (Not recommended for manual creation)

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

Advanced Tensor Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mathematical functions
   data = Tensor([[-1.0, 2.0, -3.0]])
   
   abs_result = data.abs()           # Absolute value
   exp_result = data.exp()           # Exponential
   log_result = data.abs().log()     # Natural logarithm (applied to abs for positive values)
   
   # Reduction operations
   matrix = Tensor([[1.0, 2.0], [3.0, 4.0]])
   
   sum_all = matrix.sum()                    # Sum all elements
   sum_axis0 = matrix.sum(axis=0)           # Sum along axis 0
   mean_all = matrix.mean()                 # Mean of all elements
   mean_axis1 = matrix.mean(axis=1)         # Mean along axis 1
   
   # Shape operations
   vector = Tensor([1, 2, 3, 4, 5, 6])
   reshaped = vector.reshape((2, 3))        # Reshape to 2x3 matrix

Activation Functions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from clumsygrad import activations
   
   # Create input tensor
   x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
   
   # Apply activation functions
   relu_out = activations.relu(x)        # ReLU activation
   sigmoid_out = activations.sigmoid(x)  # Sigmoid activation
   tanh_out = activations.tanh(x)        # Tanh activation
   
   # Softmax (useful for classification)
   logits = Tensor([[1.0, 2.0, 3.0]])
   softmax_out = activations.softmax(logits, axis=-1)
   
   print(f"Softmax output: {softmax_out.data}")
   print(f"Sum of softmax: {softmax_out.sum().data}")  # Should be close to 1.0

Loss Functions
~~~~~~~~~~~~~~

.. code-block:: python

   from clumsygrad import loss
   
   # Create predictions and targets
   predictions = Tensor([[0.8, 0.2, 0.9]])
   targets = Tensor([[1.0, 0.0, 1.0]], tensor_type=TensorType.INPUT)
   
   # Mean Squared Error loss
   mse = loss.mse_loss(predictions, targets)
   print(f"MSE Loss: {mse.data}")
   
   # Mean Absolute Error loss
   mae = loss.mae_loss(predictions, targets)
   print(f"MAE Loss: {mae.data}")

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

Complete Example: Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from clumsygrad.tensor import Tensor
   from clumsygrad.types import TensorType
   from clumsygrad import loss
   
   # Generate synthetic data
   np.random.seed(42)
   X_data = np.random.randn(100, 1).astype(np.float32)
   y_data = (2 * X_data + 1 + 0.1 * np.random.randn(100, 1)).astype(np.float32)
   
   # Create tensors
   X = Tensor(X_data, tensor_type=TensorType.INPUT)
   y_true = Tensor(y_data, tensor_type=TensorType.INPUT)
   
   # Initialize parameters
   W = Tensor([[1.5]], tensor_type=TensorType.PARAMETER)  # Weight
   b = Tensor([[0.0]], tensor_type=TensorType.PARAMETER)  # Bias
   
   # Training loop
   learning_rate = 0.01
   epochs = 100
   
   for epoch in range(epochs):
       # Forward pass
       y_pred = X @ W + b
       
       # Compute loss
       loss_value = loss.mse_loss(y_pred, y_true)
       
       # Backward pass
       W.grad = None  # Reset gradients
       b.grad = None
       loss_value.backward()
       
       # Update parameters
       W._data -= learning_rate * W.grad
       b._data -= learning_rate * b.grad
       
       if epoch % 20 == 0:
           print(f"Epoch {epoch}, Loss: {loss_value.data[0]:.6f}")
   
   print(f"\nFinal parameters:")
   print(f"Weight: {W.data[0][0]:.4f} (target: 2.0)")
   print(f"Bias: {b.data[0][0]:.4f} (target: 1.0)")

Key Features
~~~~~~~~~~~~

- **Automatic Differentiation**: ClumsyGrad automatically computes gradients through the computational graph
- **Tensor Types**: INPUT (data), PARAMETER (trainable), INTERMEDIATE (computed)
- **Flexible Operations**: Support for various mathematical operations and tensor manipulations
- **Activation Functions**: Built-in support for common activation functions
- **Loss Functions**: MSE and MAE loss functions for training
- **Memory Efficient**: Uses weak references to manage tensor relationships

Tips and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use appropriate tensor types**: INPUT for data, PARAMETER for trainable weights
2. **Reset gradients**: Always reset gradients before backward pass in training loops
3. **Scalar outputs for backward()**: Call backward() only on scalar tensors (typically loss values)
4. **Memory management**: The library uses weak references to prevent memory leaks in computational graphs

For more advanced usage and API documentation, see the :doc:`api_reference` section.