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

   from clumsygrad.tensor import Tensor, TensorType

   # Creating different types of tensors
   input_tensor = Tensor([1, 2, 3], tensor_type=TensorType.INPUT)
   parameter_tensor = Tensor([0.5, -0.3, 0.8], tensor_type=TensorType.PARAMETER)

Tensor Types
------------

ClumsyGrad supports three types of tensors:

**INPUT**: Data tensors that don't require gradients.
   Used for input data, targets, and constants.

**PARAMETER**: Learnable parameters that require gradients.
   Used for weights, biases, and other trainable parameters.

**INTERMEDIATE**: Temporary tensors created during operations.
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

What Automatic Differentiation Is
----------------------------------

Training a model means asking: "if I nudge this parameter slightly, how does
the output change?" That question is a derivative, and there are three ways to
answer it:

* **Numerical differentiation** approximates a derivative by evaluating the
  function at two nearby points and measuring the slope between them
  (``(f(x + h) - f(x)) / h``). It's easy to implement but only approximate,
  sensitive to the choice of ``h``, and requires re-running the whole function
  for every parameter.
* **Symbolic differentiation** derives an exact formula for the derivative,
  the way you would by hand. It's precise, but the formula can grow
  enormously for complex functions.
* **Automatic differentiation (AD)** also produces an exact derivative, but
  without ever building a symbolic formula. Instead, it breaks a computation
  into elementary steps (addition, multiplication, power, ...), and applies
  the chain rule mechanically across those steps. This is what ClumsyGrad,
  and every mainstream deep learning framework, uses.

The Computational Graph
------------------------

Every operation performed on a ``PARAMETER`` or ``INTERMEDIATE`` tensor
creates a new node that remembers two things: which tensors produced it (its
*parents*), and a *backward function* that knows how to turn a gradient
flowing into this node into gradients for its parents. This web of nodes and
connections is the **computational graph**, and it is built automatically as
your code runs; there is no separate "graph construction" step.

.. code-block:: python

   x = Tensor([2.0], tensor_type=TensorType.PARAMETER)

   u = x ** 2      # u remembers: parent=x, backward function=power_backward
   v = u + 1       # v remembers: parent=u, backward function=add_scalar_backward
   y = v ** 3      # y remembers: parent=v, backward function=power_backward

This chain, ``x`` to ``u`` to ``v`` to ``y``, is the computational graph for
this computation: data flows forward through it as each line runs, and once
``backward()`` is called, gradients flow back through the same links in the
opposite direction.

An ``INPUT`` tensor doesn't need a gradient, so operations on plain inputs
skip this bookkeeping entirely; that's why constants and data don't slow
down or clutter the graph.

The Backward Pass
------------------

Calling ``backward()`` on the final tensor (usually a scalar loss) walks the
graph in reverse: starting with a gradient of ``1`` at the output, it visits
each node in reverse order of creation, and at each one calls that node's
backward function to convert its incoming gradient into gradients for its
parents. Those parent gradients are accumulated into the parents' ``.grad``,
and the process continues until every ``PARAMETER`` tensor has received its
share.

.. code-block:: python

   # f(x) = x^2
   x = Tensor([3.0], tensor_type=TensorType.PARAMETER)
   y = x ** 2

   # Compute gradient: df/dx = 2x
   y.backward()
   print(f"x = {x.data}, gradient = {x.grad}")  # Should be 6.0

This reverse traversal is what makes AD efficient for machine learning:
computing the gradient of one scalar loss with respect to millions of
parameters takes a single backward pass, regardless of how many parameters
there are.

Chain Rule Through Multiple Operations
----------------------------------------

Because every node only needs to know how to handle its own operation, chains
of operations compose automatically; the chain rule is applied one link at a
time as ``backward()`` walks back through the graph.

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

Branching Graphs
-----------------

A tensor can be used in more than one operation, which means it can have more
than one path to the output. When that happens, gradients from every path are
added together; this is exactly what the multivariable chain rule requires.

Below, ``x`` feeds into both ``x * y`` and ``x ** 2``, so ``x`` has two paths
to ``z``. Its final gradient is the sum of the gradient contributed by each
path.

.. code-block:: python

   x = Tensor([3.0], tensor_type=TensorType.PARAMETER)
   y = Tensor([4.0], tensor_type=TensorType.PARAMETER)

   z = x * y + x ** 2   # x is used twice: once in x*y, once in x**2
   z.backward()

   print(f"dz/dx = {x.grad}")  # y + 2x = 4 + 6 = 10.0
   print(f"dz/dy = {y.grad}")  # x = 3.0

Gradient Accumulation
-----------------------

Gradients are *added* to whatever is already in ``.grad``, not overwritten.
This is what makes branching graphs work correctly above, but it also means
calling ``backward()`` more than once without clearing gradients in between
will keep piling values up:

.. code-block:: python

   x = Tensor([2.0], tensor_type=TensorType.PARAMETER)

   (x ** 2).backward()
   print(x.grad)  # [4.0]

   (x ** 2).backward()
   print(x.grad)  # [8.0]  -- accumulated, not overwritten!

   x.grad = None  # reset before computing a fresh gradient
   (x ** 2).backward()
   print(x.grad)  # [4.0]

This is why every training loop resets gradients before each backward pass,
typically with ``optimizer.zero_grad()`` (see :doc:`quickstart`).

Turning Tracking Off with ``no_grad()``
-----------------------------------------

Not every computation needs a graph. Once a model is trained, making
predictions with it doesn't require gradients at all, and building the graph
anyway would only waste memory. Wrapping code in
:func:`~clumsygrad.tensor.no_grad` disables graph construction for that
block: new tensors come out as plain ``INPUT`` tensors with
``requires_grad=False``, no matter what produced them.

.. code-block:: python

   from clumsygrad.tensor import no_grad

   x = Tensor([2.0], tensor_type=TensorType.PARAMETER)

   with no_grad():
       y = x ** 2

   print(y.requires_grad)  # False -- no graph was built for this operation

Verifying Gradients with ``gradcheck``
-----------------------------------------

Because AD relies on every operation defining a correct backward function, a
single wrong derivative can silently produce bad gradients that only show up
as a model that mysteriously fails to train. ClumsyGrad ships
:func:`~clumsygrad.gradcheck.check_gradient` to catch this: it compares the
gradient computed by ``backward()`` against a numerical approximation
(finite differences), the way you'd sanity-check a derivative by hand.

.. code-block:: python

   from clumsygrad import math
   from clumsygrad.gradcheck import check_gradient

   x = Tensor([1.0, 2.0, 3.0], tensor_type=TensorType.PARAMETER)

   passed, max_abs_diff = check_gradient(lambda t: math.sum(t ** 2), x)
   print(f"Gradients match: {passed} (max difference: {max_abs_diff:.6f})")

Differentiating a Complex Expression
--------------------------------------

ClumsyGrad isn't limited to the handful of operations a typical model uses;
it can differentiate any expression built out of its supported operations, no
matter how the trigonometric, exponential, and logarithmic pieces are nested.
Consider:

.. math::

   f(x, y) = \sin(xy) + \sqrt{e^x} - \ln(y^2 + 1)

By hand, this requires the chain rule applied through a square root, an
exponential, a logarithm, and a product rule inside a sine, twice over (once
for each variable). ClumsyGrad computes both partial derivatives with a
single ``backward()`` call:

.. code-block:: python

   from clumsygrad import math

   x = Tensor([1.5], tensor_type=TensorType.PARAMETER)
   y = Tensor([2.0], tensor_type=TensorType.PARAMETER)

   f = math.sin(x * y) + math.sqrt(math.exp(x)) - math.log(y ** 2 + 1)
   f.backward()

   print(f"f(x, y) = {f.data}")
   print(f"df/dx = {x.grad}")   # matches y*cos(xy) + 0.5*sqrt(e^x)
   print(f"df/dy = {y.grad}")   # matches x*cos(xy) - 2y/(y^2 + 1)

Every intermediate value, ``x * y``, ``exp(x)``, ``sqrt(...)``, ``y ** 2 + 1``,
and so on, is a node in the same kind of graph seen throughout this chapter.
``backward()`` doesn't know or care that this expression came from calculus
rather than a model; it just walks the graph and applies each node's backward
function. Running the same expression through
:func:`~clumsygrad.gradcheck.check_gradient` confirms both partial
derivatives match a numerical approximation.

Carry On
========

* Try the :doc:`examples` for complete, runnable training loops.
* Explore the :doc:`api_reference` for detailed function documentation.
