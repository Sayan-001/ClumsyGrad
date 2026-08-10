=========
Internals
=========

The other pages describe how to *use* ClumsyGrad. This one describes how it
*works*: the graph representation, the ``TensorType`` state machine, why
backward functions read a per-node ``extra`` dict instead of closures, and
how :meth:`~clumsygrad.tensor.Tensor.backward` walks the graph without
recursion. If you're extending the library, or just curious how a ~1000-line
autodiff engine holds together, this is the page for you.

The Computational Graph is Just Tensors
========================================

There is no separate "graph" object. A :class:`~clumsygrad.tensor.Tensor`
*is* a node: it stores a reference to its parent tensors
(``_parents: tuple[Tensor, ...]``) and the function that knows how to turn
an incoming gradient into gradients for those parents
(``_grad_fn``). Building the graph is a side effect of doing arithmetic;
every operator (``__add__``, ``__matmul__``, ``__pow__``, ...) and every
function in :mod:`clumsygrad.math` / :mod:`clumsygrad.activation` funnels
its result through the same factory, :meth:`Tensor._create_node`:

.. code-block:: python

   @staticmethod
   def _create_node(data, grad_fn, parents, extra=None) -> Tensor:
       ...
       node = Tensor(data=data, tensor_type=tensor_type)
       if tensor_type != TensorType.INPUT:
           node._grad_fn = grad_fn
           node._parents = parents
           if extra:
               node._extra.update(extra)
       ...
       return node

So ``z = x @ w + b`` doesn't build an *expression tree* that gets
interpreted later, the way a symbolic-math library would. It executes
eagerly with NumPy right away (``self._data @ other._data``) and just
*also* stashes enough breadcrumbs (``grad_fn``, ``parents``, ``extra``) to
retrace those steps backwards on demand. This is the same "define-by-run"
approach PyTorch popularized: the graph is implicit in the call history of
a normal Python program, not a data structure you construct up front.

The ``TensorType`` State Machine
=================================

Every tensor is one of three types, and the type is what decides whether an
operation is worth recording at all:

* **INPUT**: data, constants, targets. Never requires gradients on its own.
* **PARAMETER**: a leaf you intend to train. Always ``requires_grad=True``,
  and the thing optimizers actually update (see below).
* **INTERMEDIATE**: anything produced by an operation on a PARAMETER or
  another INTERMEDIATE. You're not meant to construct these directly.

The type isn't set by hand on most tensors, it's *inferred* by
``_create_node`` from the operation's inputs:

.. code-block:: python

   tensor_type = TensorType.INPUT
   if _grad_enabled:
       for parent in parents:
           if parent._tensor_type in (TensorType.PARAMETER, TensorType.INTERMEDIATE):
               tensor_type = TensorType.INTERMEDIATE
               break

In words: a result is INTERMEDIATE if *any* input to the operation is a
PARAMETER or INTERMEDIATE, and stays INPUT otherwise (e.g. two INPUT tensors
added together are still just INPUT, there's nothing to differentiate
through, so no graph bookkeeping happens for that node). ``requires_grad``
follows the same "any parent" rule (``any(parent._requires_grad for parent
in parents)``), which is what lets a gradient requirement propagate forward
through an arbitrarily long chain of operations from a single PARAMETER deep
in the graph.

This propagation is also what :func:`~clumsygrad.tensor.no_grad` hooks
into. It's a module-level flag (``_grad_enabled``), not a per-tensor
setting; while it's off, ``_create_node`` skips the parent scan entirely
and every result is a plain, ungraphed INPUT tensor, regardless of what
produced it:

.. code-block:: python

   @contextmanager
   def no_grad():
       global _grad_enabled
       previous = _grad_enabled
       _grad_enabled = False
       try:
           yield
       finally:
           _grad_enabled = previous

That's the entire mechanism: no tensor-level flags to thread through every
op, just one global checked in one place.

Why Nodes Carry an ``extra`` Dict
===================================

Most backward functions don't need ``extra`` at all. ``mul_backward``, for
instance, just reads the *other* parent's data directly:

.. code-block:: python

   def mul_backward(tensor, grad):
       x, y = tensor._parents
       return (grad * y._data, grad * x._data)

That works because both operands of ``x * y`` are themselves tensors still
sitting in ``tensor._parents``, their data is one attribute access away.
``extra`` exists for the cases where the information needed to compute a
gradient *isn't* recoverable from the parent tensors alone:

* **Scalar operations** have no tensor to read from. ``x * 2.5`` only has
  one parent (``x``); the ``2.5`` has nowhere to live except
  ``extra["scalar_value"]``, which ``mul_scalar_backward`` reads back out.
* **Broadcasting** ops (``add_broadcast_backward`` and friends) need the
  *pre-broadcast* shapes of both operands to know how to sum a gradient back
  down to size after NumPy broadcast them up for the forward pass:
  ``extra["left_shape"]`` / ``extra["right_shape"]``.
* **Reductions** (``sum``, ``mean``) need to know the *original*,
  pre-reduction shape (plus ``axis`` and ``keepdims``) to re-expand an
  incoming scalar-ish gradient back out to every element that contributed to
  it, via ``extra["input_shape"]``.
* **Indexing** (``__getitem__``) needs the original ``key`` and shape to
  scatter a gradient back into a zero tensor at the right positions via
  ``np.add.at`` (accumulating, not overwriting, when the same element was
  read more than once through repeated/fancy indices).

The common thread: ``extra`` holds whatever shape/value metadata a backward
function needs that isn't already sitting on a parent tensor. It's plain
data (a ``dict[str, Any]``) rather than, say, a closure capturing that state,
because keeping every node's bookkeeping as inspectable data instead of
opaque callables is what makes the graph-traversal utilities in
:class:`~clumsygrad.tensor.TensorUtils` and a future graph
visualizer possible.

``backward()``: Walking the Graph in Reverse
================================================

Backpropagation needs to visit every node *after* every node that depends
on it, a reverse topological order. ``backward()`` builds that order with
an iterative (stack-based) DFS rather than a recursive one, since a
recursive version would blow Python's default recursion limit on a long
training loop or any sufficiently deep graph.

The ``visited`` set in that traversal (keyed by ``Tensor._id``, a
monotonically increasing counter assigned at construction) is what keeps it
correct, and efficient, once the graph isn't a simple chain but a DAG,
e.g. the same parameter reused in two different branches that later
recombine. Without it, a shared ancestor reachable via multiple paths would
get re-expanded once per path, turning a linear traversal into one that's
exponential in graph depth for a sufficiently branchy graph.

Gradient accumulation happens for the same structural reason. Walking
``reversed(topo_order)``, each node calls its own ``_grad_fn`` and
distributes the result to its parents with ``+=``, not ``=``:

.. code-block:: python

   for node in reversed(topo_order):
       gradients = node._grad_fn(node, node._grad)
       for parent, grad in zip(node._parents, gradients):
           if parent._grad is None:
               parent._grad = grad.copy()
           else:
               parent._grad += grad

A tensor used in two places in the graph (e.g. a shared weight, or ``x``
appearing twice in ``x * x``) receives one gradient contribution per place
it was used, and those have to sum, not overwrite. This is the multivariable
chain rule showing up directly in the accumulation logic, not something
handled by any special-cased "shared node" logic elsewhere.

Cleanup
-------

Once every gradient has been distributed, ``backward()`` walks
``topo_order`` one more time and calls ``_cleanup_references()`` on every
INTERMEDIATE node, which just clears its ``_parents`` tuple (unless
``keep_graph=True`` was passed). Intermediate tensors otherwise hold strong
references back through the whole graph, which would keep every activation
from a training step alive in memory indefinitely, since nothing else
frees them. Clearing ``_parents`` after backprop breaks that chain so
Python's reference counting can reclaim them, leaving only the tensors you
actually kept a variable pointing to (typically your INPUT/PARAMETER
leaves) alive.

Keeping Memory and Latency Low
=================================

None of this is a performance library, ClumsyGrad is explicit about being
designed for educational purposes rather than speed, but a handful of
deliberate choices keep the constant factors down without complicating the
design:

* **Tensor uses** ``__slots__``. A training loop creates and discards a
  new INTERMEDIATE tensor for every single operation, so skipping the
  per-instance ``__dict__`` that a normal Python object would carry adds up
  over a graph with thousands of nodes.
* **INPUT-only computations carry zero graph overhead.** If none of an
  operation's inputs are PARAMETER or INTERMEDIATE, ``_create_node`` leaves
  the result as INPUT and never populates ``_grad_fn``, ``_parents``, or
  ``_extra`` for it at all (see the ``TensorType`` section above). Data
  preprocessing done with plain INPUT tensors costs nothing beyond the
  NumPy call itself; there's no graph being built and silently discarded.
* **no_grad() skips bookkeeping, not just backward().** Because the
  ``_grad_enabled`` check happens inside ``_create_node`` itself, every
  operation performed inside a ``no_grad()`` block is as cheap as the
  INPUT-only case above: no ``grad_fn``, no ``parents`` tuple, no ``extra``
  dict, regardless of what produced the inputs. This is why it's the
  recommended way to run inference, not just a way to save memory on the
  backward pass that never happens.
* **Backward functions reuse the forward output where it's cheap to.**
  ``exp_backward``, ``sigmoid_backward``, ``tanh_backward``,
  ``softmax_backward``, and ``relu_backward`` all read ``tensor._data``
  (the already-computed forward result) instead of recomputing
  :math:`e^x`, :math:`\sigma(x)`, etc. from the parent's input. For
  example, :math:`\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x))`, so
  ``sigmoid_backward`` reuses the forward output directly rather than
  calling the sigmoid formula a second time.
* **The visited set makes graph traversal linear, not exponential.**
  Covered above under ``backward()``: without it, a tensor reachable via
  multiple paths (a shared parameter, a value used twice) would be
  re-expanded once per path, and that blowup compounds with graph depth.
* **The graph is freed as soon as it's used.** Also covered above under
  Cleanup: ``_cleanup_references()`` drops every INTERMEDIATE tensor's
  ``_parents`` tuple right after ``backward()`` finishes, so the whole
  chain of activations from a training step can be garbage collected
  immediately rather than accumulating across iterations.
* **Gradients accumulate in place.** The ``+=`` in ``backward()`` mutates
  the existing NumPy array rather than allocating a new one on every
  contribution, which matters on graphs where a tensor is fed into many
  downstream operations.
* **Data is always float32.** Fixed at construction
  (``np.array(data, dtype=np.float32)``), this halves memory versus
  NumPy's default ``float64`` for every tensor in the graph, matching the
  default every mainstream autodiff library (PyTorch, TensorFlow, JAX)
  ships with. It's a deliberate fixed choice rather than a configurable
  ``dtype`` parameter, to avoid threading dtype-promotion rules through
  every operation for a project at this scope.
* **Optimizers do their setup once, not every step.** ``Optimizer.__init__``
  filters the parameter list down to PARAMETER tensors a single time
  (rather than re-filtering on every ``step()`` call), and
  :class:`~clumsygrad.optimizer.Adam` preallocates its first/second moment
  buffers once with ``np.zeros_like`` instead of allocating fresh arrays on
  every step.

Adding a New Operation
=========================

Every operation in the library follows the same three-part recipe, which
falls directly out of everything above:

1. **Forward**: compute the result eagerly with NumPy, then hand it to
   ``Tensor._create_node(data=..., grad_fn=..., parents=(...), extra=...)``.
   Only pass ``extra`` if the backward function needs shape/value
   information that isn't already reachable from ``parents``.
2. **Backward**: a function ``(tensor, grad) -> GradientTuple`` in
   :mod:`clumsygrad.grad`, returning one gradient array per entry in
   ``parents``, in the same order.
3. **Verify**: a numerical check via :func:`~clumsygrad.gradcheck.check_gradient`,
   which compares your analytical ``grad_fn`` against a finite-difference
   approximation. This is the fastest way to catch a sign error or a
   forgotten broadcast reduction; see :mod:`clumsygrad.gradcheck` for
   details.

Carry On
========

* Read the :doc:`tutorial` for how to *use* tensors, gradients, and the graph.
* Browse the :doc:`api_reference` for full function documentation.
