.. ClumsyGrad documentation master file

==========
ClumsyGrad
==========

**A Simple Automatic Differentiation Library for Educational Purposes**

Overview
========

ClumsyGrad is a lightweight automatic differentiation library built on top of NumPy, designed specifically for educational purposes. It provides a simple yet powerful framework for understanding the fundamentals of:

* **Tensor Operations**: Basic mathematical operations on multi-dimensional arrays
* **Automatic Differentiation**: Forward and backward propagation for gradient computation
* **Computational Graphs**: Building and traversing computation graphs

Key Features
============

* **Lightweight Design**
   Minimal dependencies - built primarily on NumPy

* **Extensible Architecture**
   Easy to understand and modify for experimentation

* **Complete Gradient Support**
   Full automatic differentiation with computational graph tracking


Quick Start Example
===================

Here's a simple example demonstrating ClumsyGrad's capabilities:

.. code-block:: python

   from clumsygrad.tensor import Tensor
   from clumsygrad.types import TensorType
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

.. code-block:: bash

   pip install clumsygrad


.. note::
   ClumsyGrad requires Python 3.9+ and NumPy. No other dependencies are needed for core functionality.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :numbered:

   quickstart
   tutorial
   api_reference

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   clumsygrad.tensor
   clumsygrad.types
   clumsygrad.grad
   clumsygrad.loss
   clumsygrad.optimizer
   clumsygrad.activation
   clumsygrad.random

Performance Notes
=================

.. warning::
   ClumsyGrad is designed for educational purposes. For production workloads, 
   consider using established frameworks like PyTorch or TensorFlow.

* **Memory Management**
   The library uses weak references to prevent memory leaks in computational graphs.

* **Computational Efficiency** 
   Operations are implemented using NumPy for reasonable performance on CPU.

* **Gradient Computation**
   Automatic differentiation is implemented using reverse-mode (backpropagation).

Contributing
============

ClumsyGrad welcomes contributions! Whether you're:

* **Reporting bugs**
* **Suggesting features** 
* **Improving documentation**
* **Contributing code**

Acknowledgments
===============

* Inspired by PyTorch and other automatic differentiation frameworks.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`