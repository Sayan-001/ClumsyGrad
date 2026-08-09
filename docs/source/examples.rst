========
Examples
========

Complete examples are listed below which show the library in action.

Example 1: Linear Regression
=============================

Fits a line to noisy data using a single weight matrix and bias, trained with
:class:`~clumsygrad.optimizer.SGD`.

.. code-block:: python

   from clumsygrad.loss import mse_loss
   from clumsygrad.optimizer import SGD
   from clumsygrad.random import randn
   from clumsygrad.tensor import Tensor, TensorType

   # Data following y = 3*x1 - 2*x2 + 5, with a little noise
   X = randn((100, 2), tensor_type=TensorType.INPUT)
   true_W = Tensor([[3.0], [-2.0]])
   true_b = Tensor([5.0])
   y = X @ true_W + true_b + randn((100, 1)) * 0.1

   # The model to train
   W = Tensor([[0.0], [0.0]], tensor_type=TensorType.PARAMETER)
   b = Tensor([0.0], tensor_type=TensorType.PARAMETER)
   optimizer = SGD([W, b], lr=0.1)

   for epoch in range(100):
       optimizer.zero_grad()
       prediction = X @ W + b
       loss = mse_loss(prediction, y)
       loss.backward()
       optimizer.step()

   print(f"Loss: {loss.data:.4f}")
   print(f"Learned W: {W.data.flatten()}, b: {b.data.flatten()}")

Example 2: Logistic Regression
================================

Passing a linear model through :func:`~clumsygrad.activation.sigmoid` turns it
into a binary classifier. This example also swaps in
:class:`~clumsygrad.optimizer.Adam`, which can be used anywhere
:class:`~clumsygrad.optimizer.SGD` is used.

.. code-block:: python

   from clumsygrad import activation as act
   from clumsygrad.loss import mse_loss
   from clumsygrad.optimizer import Adam
   from clumsygrad.random import randn
   from clumsygrad.tensor import Tensor, TensorType

   # Points labeled by which side of a line they fall on
   X = randn((100, 2), tensor_type=TensorType.INPUT) * 2
   true_W = Tensor([[4.0], [4.0]])
   target = act.sigmoid(X @ true_W)

   W = Tensor([[0.0], [0.0]], tensor_type=TensorType.PARAMETER)
   b = Tensor([0.0], tensor_type=TensorType.PARAMETER)
   optimizer = Adam([W, b], lr=0.1)

   for epoch in range(150):
       optimizer.zero_grad()
       prediction = act.sigmoid(X @ W + b)
       loss = mse_loss(prediction, target)
       loss.backward()
       optimizer.step()

   print(f"Loss: {loss.data:.4f}")

Example 3: A Small Neural Network
===================================

A single hidden layer with a :func:`~clumsygrad.activation.relu` activation
can fit curves a straight line cannot, such as ``y = x^2``.

.. code-block:: python

   from clumsygrad import activation as act
   from clumsygrad.loss import mse_loss
   from clumsygrad.optimizer import Adam
   from clumsygrad.random import rand, randn
   from clumsygrad.tensor import Tensor, TensorType

   X = rand((100, 1), tensor_type=TensorType.INPUT) * 4 - 2  # range: -2 to 2
   y = X**2

   w1 = randn((1, 10), tensor_type=TensorType.PARAMETER)
   b1 = Tensor([[0.0] * 10], tensor_type=TensorType.PARAMETER)
   w2 = randn((10, 1), tensor_type=TensorType.PARAMETER)
   b2 = Tensor([0.0], tensor_type=TensorType.PARAMETER)
   optimizer = Adam([w1, b1, w2, b2], lr=0.05)

   for epoch in range(300):
       optimizer.zero_grad()
       hidden = act.relu(X @ w1 + b1)
       prediction = hidden @ w2 + b2
       loss = mse_loss(prediction, y)
       loss.backward()
       optimizer.step()

   print(f"Loss: {loss.data:.4f}")

Example 4: Inference with ``no_grad()``
=========================================

Making predictions with a trained model doesn't require building a
computational graph. Wrapping the forward pass in
:func:`~clumsygrad.tensor.no_grad` skips graph tracking, so the result comes
back as a plain tensor with no gradient bookkeeping attached.

.. code-block:: python

   from clumsygrad.tensor import no_grad

   # Continuing from the model trained in Example 3
   with no_grad():
       hidden = act.relu(X @ w1 + b1)
       prediction = hidden @ w2 + b2

   print(prediction.requires_grad)  # False
   print(prediction.data[:3].flatten())
   print(y.data[:3].flatten())

Carry On
========

* Read the :doc:`tutorial` for a step-by-step introduction to tensors and gradients.
* Browse the :doc:`api_reference` for full function documentation.
