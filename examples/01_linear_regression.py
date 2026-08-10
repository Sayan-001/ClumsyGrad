# Problem being solved: recover the coefficients of a linear relationship
# (y = 3*x1 - 2*x2 + 5) from noisy observations.

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
