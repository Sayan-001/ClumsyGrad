.. ClumsyGrad documentation master file

==========
ClumsyGrad
==========

**A Simple Automatic Differentiation Library built on top of NumPy**

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :numbered:

   quickstart
   tutorial
   api_reference

Key Features
============

* **Lightweight Design**
   Minimal dependencies - built primarily on NumPy

* **Minimalist Architecture**
   Easy to understand and modify for experimentation

Quick Start Example
===================

.. code-block:: python

   from clumsygrad.tensor import Tensor, TensorType
   from clumsygrad.loss import mse_loss
   
   # Create tensors with gradient tracking
   x = Tensor([[1.0, 2.0]], tensor_type=TensorType.INPUT)
   W = Tensor([[3.0], [4.0]], tensor_type=TensorType.PARAMETER)
   b = Tensor([5.0], tensor_type=TensorType.PARAMETER)
   target = Tensor([15.0], tensor_type=TensorType.INPUT)
   
   # Forward pass: linear transformation
   output = x @ W + b  # Matrix multiplication + bias
   
   # Compute loss
   loss = mse_loss(output, target)
   
   # Backward pass: automatic differentiation
   loss.backward()
   
   # Gradients are now available
   print(f"W gradient: {W.grad}")
   print(f"b gradient: {b.grad}")
   print(f"Loss: {loss.data}")

Installation
============

To install ClumsyGrad, you can use pip:

.. code-block:: bash

   pip install clumsygrad


.. note::
   ClumsyGrad requires Python 3.9+ and NumPy.

Performance Notes
=================

.. warning::
   ClumsyGrad is mainly designed by me for educational purposes, and may not be completely optimzed for performance.

Contributing
============

ClumsyGrad welcomes contributions! Whether you're:

* **Reporting bugs**
* **Suggesting features** 
* **Improving documentation**
* **Contributing code**

Acknowledgments
===============

* Inspired by PyTorch (obviously!) and other automatic differentiation frameworks.